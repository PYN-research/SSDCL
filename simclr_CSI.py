import time

import torch.optim
import numpy as np
import transform_layers as TL
from contrastive_loss import get_similarity_matrix, NT_xent
from utils import AverageMeter, normalize
from gmm2 import GMM
from ood_pre import get_features
from Kmeans_initial1 import KMeans_init
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
n_rots=4
ndf=64
num_mixtures=2
momentum=0.999
def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_unsup_{mode}'

    if mode == 'simclr':
        from simclr import train
    elif mode == 'simclr_CSI':
        from simclr_CSI import train
        fname += f'_shift_{P.shift_trans_type}'
    else:
        raise NotImplementedError()

    if P.one_class_idx is not None:
        fname += f'_one_class_{P.one_class_idx}'

    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname


def update_comp_loss(loss_dict, loss_in, loss_out, loss_diff, batch_size):
    loss_dict['pos'].update(loss_in, batch_size)
    loss_dict['neg'].update(loss_out, batch_size)
    loss_dict['diff'].update(loss_diff, batch_size)


def summary_comp_loss(logger, tag, loss_dict, epoch):
    logger.scalar_summary(f'{tag}/pos', loss_dict['pos'].average, epoch)
    logger.scalar_summary(f'{tag}/neg', loss_dict['neg'].average, epoch)
    logger.scalar_summary(f'{tag}', loss_dict['diff'].average, epoch)
def normalized(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
def Init_gmm_parameters(x_train):
    x_train = torch.reshape(x_train, (n_rots, len(x_train)//n_rots, ndf))
    x_train = x_train.transpose(1,0)
    state = KMeans_init(x_train,num_mixtures,n_rots,ndf)
    means_init,pis_init,cov_init,S1_init,S2_init,S3_init = state
    return means_init,pis_init,cov_init,S1_init,S2_init,S3_init
def tc_loss(zs, m):
    zs = torch.reshape(zs, (n_rots, len(zs)//n_rots, -1))
    zs = zs.transpose(1,0)
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss
def pretrain(P, epoch, model, loader,simclr_aug=None, linear=None, linear_optim=None):
    assert simclr_aug is not None
    assert P.K_shift > 1
    #get initial features    
    Label_all=[]
    feature_all = torch.zeros((20000, ndf))
    for n, (images, labels) in enumerate(loader):
        model.eval()
        batch_size = images.size(0)
        images = images.to(device)
        print('images_shape:',images.shape)
        images = torch.cat([P.shift_trans(images, k) for k in range(P.K_shift)])
        print('images_type:',images.dtype)
        images = simclr_aug(images)  # transform (4B)
        _, outputs_aux = model(images, simclr=True, penultimate=True, shift=True, em_unfolding=True)
        idxx=np.arange(4*batch_size)+(256*n)
        feature_all[idxx] = outputs_aux['em_unfolding'].data.cpu()  
        Label_all.append(labels)
    #get initialization of gmm
  #  feature_all=torch.nn.functional.normalize(feature_all, p=2, dim=1, eps=1e-12)
    print('Feature_all_shape:',feature_all.shape)
    means,pis,cov,S1,S2,S3 = Init_gmm_parameters(feature_all)
    state=means,pis,cov,S1,S2,S3
    return state

def train(P, epoch, model, criterion, optimizer, scheduler, loader, state, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_shift > 1
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    means,pis,cov,S1,S2,S3=state
    #training    
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
        images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)

        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True, em_unfolding=True)
        #Sim loss
        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda
        #Classifier loss
        loss_shift = criterion(outputs_aux['shift'], shift_labels)
        #clustering loss
        #EM-block
        gmm_block=GMM(num_mixtures, ndf,n_rots,batch_size,momentum,means,pis,cov,S1,S2,S3)
     #   norm_feature=torch.nn.functional.normalize(outputs_aux['em_unfolding'], p=2, dim=1, eps=1e-12)
        norm_feature=outputs_aux['em_unfolding']
        gmm_nce_loss,S1_up,S2_up,S3_up,mu_up,pis_up,cov_up,gamma_up=gmm_block(norm_feature[:(4*batch_size),])#(4B,128)
        S1=S1_up
        S2=S2_up
        S3=S3_up
        means=mu_up
        pis=pis_up
        cov=cov_up
        ### total loss ###
        loss = loss_sim +loss_shift + gmm_nce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        penul_1 = outputs_aux['penultimate'][:batch_size]#(B,C)
        penul_2 = outputs_aux['penultimate'][P.K_shift * batch_size: (P.K_shift + 1) * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation
        
        
        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['cls'].update(gmm_nce_loss.item(), batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))
    
    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average, losses['shift'].average))
    means_final = means
    pis_final = pis
    cov_final = cov
    state = means_final, pis_up, cov_up, S1, S2, S3
    print('mu_final:',(means_final.min(),means_final.max()))
    print('pis_final:',(pis_final.min(),pis_final.max()))
    print('cov_final:',(cov_final.min(),cov_final.max()))
    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
    return means_final,pis_final,cov_final,state
