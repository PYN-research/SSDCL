import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from classifier import get_simclr_augmentation
import transform_layers as TL
from utils import set_random_seed, normalize
from evals import get_auroc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
n_rots=4


def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)
    
    P.axis = []
    for f in feats_train['simclr'].chunk(P.K_shift, dim=1):
        #print('f_shape:',f.shape)#(5000,10,128)
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))#(5000,)

    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)
    
    weight_sim = []#(4,)
    weight_shi = []#(4,)
    for shi in range(P.K_shift):
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())
    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
        
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
        
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))
    #print(f'weight_pen:\t' + '\t'.join(map('{:.4f}'.format, P.weight_pen)))
    print('Pre-compute features...')
    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    #T-SNE
    #print('In-distribution_TSNE:')
    #test_digits(label_id_gmm,feats_id_gmm)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)
            
    print(f'Compute OOD scores... (score: {ood_score})')
    
    
    
    #compute scores of normal samples
    scores_id,score_add = get_scores(P, feats_id, ood_score)#(1000,)
    print('id_CSI_score:',(torch.tensor(scores_id).min(),torch.tensor(scores_id).max()))
    print('id_Cluster_score:',(torch.tensor(score_add).min(),torch.tensor(score_add).max()))
    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []
    #compute scores of abnormal samples
    for ood, feats in feats_ood.items():
        scores_ood[ood],score_ood = get_scores(P, feats, ood_score)
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])#(9000,)
    print('ood_CSI_score:',(torch.tensor(scores_ood[ood]).min(),torch.tensor(scores_ood[ood]).max()))
    print('ood_Cluster_score:',(torch.tensor(score_ood).min(),torch.tensor(score_ood).max()))
    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict

def cosine_similarity(x1 ,x2):
    eps = torch.autograd.Variable(torch.FloatTensor([1.e-8]), requires_grad=False)
    dot_prod = torch.matmul(x1,x2.transpose(2,1))
    dist_x1 = torch.norm(x1, p=2, dim=-1, keepdim=True) 
    dist_x2 = torch.norm(x2, p=2, dim=-1, keepdim=True)
    return dot_prod / torch.max(dist_x1*dist_x2, eps.cuda())

def Cluster_score(samples,mu_up):
    out_values = []
    for sample in samples:
        mu=mu_up.squeeze(0).float().cuda()
        mu_norm=torch.norm(mu,p=2,dim=-1,keepdim=True)
        mu=mu/mu_norm
        diff = torch.matmul(sample.float(),mu.transpose(2,1))
        diff,_ = torch.max(diff,0)
        out_values.append(diff.data.cpu().numpy())
        
    out1 = torch.FloatTensor(out_values).cuda()
    out1 = torch.diagonal(out1, dim1=1, dim2=2)
    out1 = out1.sum(1)
    out1 = out1.cpu().data.numpy()
    return out1


def M_score(x,mu,cov):
    diff=x-mu.unsqueeze(0).float().cuda()
    diff=diff.unsqueeze(2)
    operator=torch.matmul(diff,cov.unsqueeze(0).float().cuda())
    diff1=(operator*diff).squeeze(2)
    diff1=diff1.sum(2)
    diff1,_=torch.max(-diff1,dim=1)
    diff1=diff1.cpu().data.numpy()
    return diff1
def get_scores(P, feats_dict, ood_score):
    # convert to gpu tensor
    n_rots=4
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    feats_pem = feats_dict['em_unfolding'].to(device)
    feats_pem = [feats_pem.mean(dim=1) for f in feats_pem.chunk(P.K_shift, dim=1)]
    feats_pem = torch.cat((feats_pem[0],feats_pem[1],feats_pem[2],feats_pem[3]))
  
    N = feats_sim.size(0)
    # get centers
    feats_pem = torch.reshape(feats_pem, (n_rots, len(feats_pem)//n_rots, 64))
    feats_pem = feats_pem.transpose(1,0)#(1000,4,64)
  #  feats_pem = torch.nn.functional.normalize(feats_pem, p=2, dim=2, eps=1e-12)
    means = torch.load("./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/means.pt".format(P.one_class_idx))
    pis = torch.load("./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/pis.pt".format(P.one_class_idx))
    cov = torch.load("./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/cov.pt".format(P.one_class_idx))
    
    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        
        score = 0
        for shi in range(P.K_shift):
            score += (f_sim[shi] * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
            score += f_shi[shi][:, shi].item() * P.weight_shi[shi]
            
        score = score / P.K_shift
        scores.append(score)
    score_add = Cluster_score(feats_pem, means)#
    scores = torch.tensor(scores)+torch.tensor(0.1*score_add)
    
    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores,score_add



def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift','penultimate')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    for layer in layers:
         path = prefix + f'_{data_name}_{layer}.pth'
         if os.path.exists(path):
             feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value
    #feature_vis=feature_visual
    return feats_dict

def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift','penultimate')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    #out_feats=[]
    #label_rot=[]
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)
            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)
                

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()#(256,64)
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor
      
        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)
                

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]
        
    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

