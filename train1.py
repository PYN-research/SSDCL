from utils import Logger
from utils import save_checkpoint
from utils import save_linear_checkpoint

from train import *
from evals import test_classifier
from simclr_CSI import pretrain

from simclr_CSI import setup
train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Run experiments
state_new=[]
for epoch in range(0, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {}
    kwargs['linear'] = linear
    kwargs['linear_optim'] = linear_optim
    kwargs['simclr_aug'] = simclr_aug
    print("Start pretrainig:")
    if epoch==0:
        state = pretrain(P, epoch, model, train_loader, **kwargs)
    else:
        state = state_new
    print("Start training:")
    means_final, pis_final, cov_final, state_new = train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, state, logger=logger, **kwargs)

    model.eval()

    if epoch % P.save_step == 0 and P.local_rank == 0:
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)
        torch.save(means_final,"./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/means.pt".format(P.one_class_idx))
        torch.save(pis_final,"./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/pis.pt".format(P.one_class_idx))
        torch.save(cov_final,"./logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_{}/cov.pt".format(P.one_class_idx))
    if epoch % P.error_step == 0 and ('sup' in P.mode):
        error = test_classifier(P, model, test_loader, epoch, logger=logger)

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_error', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))
