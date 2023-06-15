import clamp
from clamp.deeptox.monitor import MetricMonitor
from clamp.deeptox.loghelper import LogHelper
from clamp.deeptox import metrics

from numpy.random import default_rng
from torch.utils.data import Subset, RandomSampler, SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import MultiplicativeLR
from scipy import sparse
from scipy.special import expit as sigmoid

from loguru import logger
from pathlib import Path
from typing import Optional

from . import dataset
from .models import models, scaled, multitask, gnn, pretrained

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import mlflow
import wandb
import argparse
from tqdm import tqdm
import os

def parse_hidden_layers(s:str):
    """Parse a string in the form of [32, 32] into a list of integers."""
    try:
        res = [int(ls) for ls in s.strip('[]').split(',')]
    except:
        raise argparse.ArgumentTypeError('String in the form of [32, 32] expected for hidden_layers')
    return res

LOGGER_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name: <19}</cyan> | <cyan>{line: >3}</cyan> | <cyan>{function: <30}</cyan> | <level>{message}</level>'
NAME2FORMATTER = {
    'assay_mode': str,
    'model': str,
    'multitask_temperature': float,
    'optimizer': str,
    'hidden_layers': parse_hidden_layers,
    'compound_layer_sizes': parse_hidden_layers,
    'assay_layer_sizes': parse_hidden_layers,
    'embedding_size': int,
    'lr_ini': float,
    'epoch_max': int,
    'batch_size': int,
    'dropout_input': float,
    'dropout_hidden': float,
    'l2': float,
    'nonlinearity': str,
    'pooling_mode': str,
    'lr_factor': float,
    'patience': int,
    'attempts': int,
    'loss_fun': str,
    'tokenizer':str,
    'transformer':str,
    'warmup_epochs':int,
    'train_balanced':int,
    'beta':float,
    'norm':bool,
    'label_smoothing':float,
    'gpu':int,
    'checkpoint':str,
    'verbose':bool,
    'hyperparams':str,
    'format':str,
    'f':str,
    'support_set_size':int,
    'train_only_actives':bool,
    'random':int,
    'seed':int,
    'dataset':str,
    'experiment':float,
    'split':str,
    'wandb':str,
    'compound_mode':str,
    'train_subsample':float,
}

EVERY = 50000

def interupptable(func):
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.error(f'{func.__name__} manually interrupted.')
    return wrap

def seed_everything(seed=70135):
    """ does what it says ;) - from https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335"""
    import numpy as np
    import random
    import os
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def init_model(
        compound_features_size: int,
        assay_features_size: int,
        hp: dict,
        verbose: bool = False
) -> models.DotProduct:
    """
    Initialize PyTorch model.

    Parameters
    ----------
    compound_features_size: int
        Input size of the compound encoder.
    assay_features_size: int
        Input size of the assay encoder.
    hp: dict
        Hyperparameters.
    verbose: bool
        Be verbose if True.

    Returns
    -------
    :class:`DotProduct`
        Model instance.
    """
    model_str = hp['model']
    modes = ['Multitask', 'Scaled', 'GNN', 'Pretrained']
    selected_mode = ''
    for mode in modes:
        if mode in model_str:
            model_str = model_str.replace(mode,'')
            selected_mode = mode

    #if not (hasattr(models,model_str) or hasattr(gnn, 'GNN'+model_str)): 
    #    raise NotImplementedError(f'Model "{hp["model"]}" is not known.')

    # ['Linear',', 'MLPLayerNorm',', 'ScaledMLPLayerNorm',', 'MultitaskMLPLayerNorm',

    if verbose:
        logger.info(f'Initialize "{hp["model"]}" model.')

    init_dict = hp.copy()

    init_dict.pop('embedding_size') # hast to be provided as positional argument
    
    
    logger.info(selected_mode+' has been selected')
    #TODO remove ifs and replace with getattr

    model = getattr(clamp.models, hp['model'])(
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        embedding_size = hp['embedding_size'],
        **init_dict)

    if wandb.run:
        wandb.watch(model, log_freq=100, log_graph=(True))
    return model


def train_and_test(
        biobert: dataset.InMemoryClamp,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        test_idx: np.ndarray,
        hparams: dict,
        run_info: mlflow.entities.RunInfo,
        checkpoint_file: Optional[Path] = None,
        keep: bool = True,
        device: str = 'cpu',
        bf16: bool = False,
        verbose: bool = True
) -> None:
    """
    Train a model on `biobert[train_idx]` while validating on
    `biobert[valid_idx]`. Once training is completed, evaluate the model
    on `biobert[test_idx]`.

    A model-optimizer PyTorch checkpoint can be passed to resume training.

    Parameters
    ----------
    biobert: :class:`~biobert.dataset.ImBioBert`
        Dataset instance.
    train_idx: :class:`numpy.ndarray`
        Activity indices of the training split.
    valid_idx: :class:`numpy.ndarray`
        Activity indices of the validation split.
    test_idx: :class:`numpy.ndarray`
        Activity indices of the test split.
    hparams: dict
        Model characteristics and training strategy.
    run_info: :class:`mlflow.entities.RunInfo`
        MLflow's run details (for logging purposes).
    checkpoint_file: str or :class:`pathlib.Path`
        Path to a model-optimizer checkpoint from which to resume training.
    keep: bool
        Keep the persisted model weights if True, remove them otherwise.
    device: str
        Computing device.
    verbose: bool
        Be verbose if True.
    """

    if verbose:
        if checkpoint_file is None:
            message = 'Start training.'
        else:
            message = f'Resume training from {checkpoint_file}.'
        logger.info(message)

    # set up call-specific logging (run)
    loghelper = LogHelper(run_info.experiment_id, run_info.run_id, logger_format=LOGGER_FORMAT)
    loghelper.start()

    # initialize checkpoint, if any
    # (if no checkpoint is given, an empty dict is returned)
    checkpoint = init_checkpoint(checkpoint_file, device)

    # set up metric monitoring
    metric_monitor = MetricMonitor(
        'davgp', #valid_mean_davgp
        value=checkpoint.get('value', None),
        epoch=checkpoint.get('epoch', 0),
        lower_is_better=False,
        patience=hparams['patience'],
        attempts=hparams['attempts'],
        verbose=verbose
    )

    # initialize model
    print(hparams)
    if 'Multitask' in hparams.get('model'):

        _, train_assays = biobert.get_unique_names(train_idx)
        biobert.setup_assay_onehot(size=train_assays.index.max() + 1)
        train_assay_features = biobert.assay_features[:train_assays.index.max() + 1]
        train_assay_features_norm = F.normalize(
            torch.from_numpy(train_assay_features),
            p=2, dim=1
        ).to(device)

        model = init_model(
            compound_features_size=biobert.compound_features_size,
            assay_features_size=biobert.assay_onehot.size,
            hp=hparams,
            verbose=verbose
        )

    else:
        model = init_model(
            compound_features_size=biobert.compound_features_size,
            assay_features_size=biobert.assay_features_size,
            hp=hparams,
            verbose=verbose
        )

    if 'model_state_dict' in checkpoint:
        if verbose:
            logger.info('Load model state_dict from checkpoint into model.')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()

    # assignment is not necessary when moving modules, but it is for tensors
    # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
    # here I only assign for consistency
    model = model.to(device)

    # initialize optimizer
    # https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer
    # Moving a model to the GPU should be done before the creation of its
    # optimizer. I think I do it right here
    optimizer = init_optimizer(model, hparams, verbose)

    if 'optimizer_state_dict' in checkpoint:
        if verbose:
            logger.info('Load optimizer state_dict from checkpoint into optimizer.')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    if 'loss_fun' in hparams:
        class CustomCE(nn.CrossEntropyLoss):
            """ Cross entropy loss #TODO doc"""
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                beta = 1/(input.shape[0]**(1/2))
                input = input*(target*2-1)*beta # target from [0,1] to [-1,1]
                target = torch.arange(0,len(input)).to(input.device)
                
                return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        class ConLoss(nn.CrossEntropyLoss):
            """"Contrastive Loss"""
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                sigma = 1
                bs = target.shape[0]
                #only modify diag that is a negative
                # eg makes this from a target of [0, 1, 0]
                #tensor([[ -1.,  1.,  1.],
                #        [  1.,  1.,  1.],
                #        [  1.,  1., -1.]])
                modif = (1-torch.eye(bs)).to(target.device) + (torch.eye(bs).to(target.device)*(target*2-1)) 
                input = input*modif/sigma

                diag_idx = torch.arange(0,len(input)).to(input.device)

                label_smoothing = hparams.get('label_smoothing', 0.0)
                if label_smoothing is None: #if it's in label_smoothing but still None
                    label_smoothing = 0.0

                mol2txt = F.cross_entropy(input,   diag_idx, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)
                txt2mol = F.cross_entropy(input.T, diag_idx, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)
                return mol2txt+txt2mol

        str2loss_fun = {
            'BCE': nn.BCEWithLogitsLoss(),
            'CE': CustomCE(),
            'Con': ConLoss(),
        }
        assert hparams['loss_fun'] in str2loss_fun ,\
            "loss_fun not implemented"
        criterion = str2loss_fun[hparams['loss_fun']]

    criterion = criterion.to(device)

    # set up learning rate scheduler
    # lambda function below returns `lr_factor` whatever the input to lambda is
    if 'lr_factor' in hparams:
        lr_factor = hparams['lr_factor']
    else:
        lr_factor = 1
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda _: lr_factor)

    #lot_lr_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=1000,eta_min=0)
    num_steps_per_epoch = len(train_idx)/hparams['batch_size']
    class Linwarmup():
        def __init__(self, steps=10000):
            self.step = 0
            self.max_step = steps
            self.step_size = 1/steps
        def get_lr(self, lr):
            if self.step>self.max_step: 
                return 1
            new_lr = self.step*self.step_size
            self.step +=1
            return new_lr

    #TODO Bug when set to 0
    if hparams.get('warmup_steps'):
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=Linwarmup(steps=num_steps_per_epoch*hparams.get('warmup_epochs',0)).get_lr)
    else:
        scheduler2 = None

    if lr_factor != 1:
        logger.info(f'Scheduler enabled with lr_factor={hparams["lr_factor"]}. Note that this makes different runs difficult to compare.')
    else:
        logger.info('Scheduler enabled with lr_factor=1. This keeps the interface but results in no reduction.')

    # set up batch samplers for the indices;
    # the actual dataset slicing is actually done manually
    train_sampler = RandomSampler(data_source=train_idx)

    valid_sampler = SequentialSampler(data_source=valid_idx)
    valid_batcher = BatchSampler(
        sampler=valid_sampler,
        batch_size=hparams['batch_size'],
        drop_last=False
    )

    test_sampler = SequentialSampler(data_source=test_idx)
    test_batcher = BatchSampler(
        sampler=test_sampler,
        batch_size=hparams['batch_size'],
        drop_last=False
    )

    epoch = checkpoint.get('epoch', 0)
    new_train_idx = None
    while epoch < checkpoint.get('epoch', 0) + hparams['epoch_max']:
        if hparams.get('train_balanced',False):
            logger.info('sampling balanced')
            num_pos = biobert.activity.data[train_idx].sum()
            # tooo large with WeightedRandomSampler
            #num_neg = (len(train_idx))-num_pos
            remove_those = train_idx[((biobert.activity.data[train_idx]) == 0)]
            remove_those = np.random.choice(remove_those, size=int(len(remove_those)-num_pos))
            idx = np.in1d(train_idx, remove_those)
            new_train_idx = train_idx[~idx]
            if isinstance(hparams['train_balanced'], int):
                max_samples_per_epoch = hparams['train_balanced']
                if max_samples_per_epoch>1:
                    logger.info(f'using only {max_samples_per_epoch} for one epoch')
                    new_train_idx = np.random.choice(new_train_idx, size=max_samples_per_epoch)
            train_sampler = RandomSampler(data_source=new_train_idx)
        if hparams.get('train_subsample',0)>0:
            if hparams['train_subsample']<1:
                logger.info(f'subsample training set to {hparams["train_subsample"]*100}%')
                hparams['train_subsample'] = int(hparams['train_subsample']*len(train_idx))
            logger.info(f'subsample training set to {hparams["train_subsample"]}')
            sub_train_idx = np.random.choice(train_idx if new_train_idx is None else new_train_idx, size=int(hparams['train_subsample']))
            train_sampler = RandomSampler(data_source=sub_train_idx)

        train_batcher = BatchSampler(
            sampler=train_sampler,
            batch_size=hparams['batch_size'],
            drop_last=False
        )

        # train
        loss_sum = 0.
        preactivations_l = []
        topk_l, arocc_l = [], []
        activity_idx_l = []
        for nb, batch_indices in enumerate(train_batcher): #tqdm(,mininterval=2)

            # get and unpack batch data
            batch_data = Subset(biobert, indices=train_idx)[batch_indices]
            activity_idx, compound_features, assay_features, assay_onehot, activity = batch_data

            # move data to device
            # assignment is not necessary for modules but it is for tensors
            # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
            assay_onehot = assay_onehot.to(device).float() if not isinstance(assay_onehot[0], str) else assay_onehot
            activity = activity.to(device)

            # forward
            with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float32):
                if hparams.get('loss_fun') in ('CE', 'Con'):
                    preactivations = model.forward_dense(compound_features, 
                        assay_onehot if 'Multitask' in hparams['model'] else assay_features)
                else:
                    preactivations = model(compound_features, 
                        assay_onehot if 'Multitask' in hparams['model'] else assay_features)

                # loss
                beta = hparams.get('beta',1) 
                if beta is None: beta = 1
                preactivations = preactivations*1/beta
                loss = criterion(preactivations, activity)
            

            # zero gradients, backpropagation, update
            optimizer.zero_grad()
            loss.backward()
            if hparams.get('optimizer')=='SAM':
                def closure():
                    preactivations = model(compound_features, 
                        assay_onehot if 'Multitask' in hparams['model'] else assay_features)
                    loss = criterion(preactivations, activity)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.step()
                scheduler.step()
                if scheduler2: scheduler2.step()

            # accumulate loss
            loss_sum += loss.item()
            
            if hparams.get('loss_fun')=='Con':
                from mhnreact.utils import top_k_accuracy
                ks = [1,5,10,50]
                tkaccs, arocc = top_k_accuracy(torch.arange(0,len(preactivations)), preactivations, k=[1,5,10,50], ret_arocc=True)
                topk_l.append(tkaccs) #allready detached numpy ;)
                arocc_l.append(arocc)

            if hparams.get('loss_fun') in ('CE','Con'):
                #preactivations = preactivations.sum(axis=1)
                preactivations = torch.diag(preactivations) #get only diag elements

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad is True
            # - move it to cpu
            preactivations_l.append(preactivations.detach().cpu())

            # accumulate indices to track order in which the dataset is visited
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx)

            if nb % EVERY == 0 and verbose:
                logger.info(f'Epoch {epoch}: Training batch {nb} out of {len(train_batcher) - 1}.')

        # log mean loss over all minibatches
        mlflow.log_metric('train_loss', loss_sum / len(train_batcher), step=epoch)
        if wandb.run: 
            wandb.log({
                'train/loss':loss_sum / len(train_batcher), 
                'lr':scheduler2.get_last_lr()[0] if scheduler2 else scheduler.get_last_lr()[0]
                    }, step=epoch) 


        # compute metrics for each assay (on the cpu)
        preactivations = torch.cat(preactivations_l, dim=0)
        probabilities = torch.sigmoid(preactivations).numpy()

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        #assert np.array_equal(np.sort(activity_idx), train_idx)
        #assert not np.array_equal(activity_idx, train_idx)

        targets = sparse.csc_matrix(
            (
                biobert.activity.data[activity_idx],
                (
                    biobert.activity.row[activity_idx],
                    biobert.activity.col[activity_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.bool
        )

        scores = sparse.csc_matrix(
            (
                probabilities,
                (
                    biobert.activity.row[activity_idx],
                    biobert.activity.col[activity_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.float32
        )


        md = metrics.swipe_threshold_sparse(
            targets=targets,
            scores=scores, verbose=verbose>=2, ret_dict=True
        ) #returns dict for with metric per assay in the form of {metric: {assay_nr: value}}

        if hparams.get('loss_fun')=='Con':
            for ii, k in enumerate(ks):
                md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()}  #drop last (might be not full)
            md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} #drop last (might be not full)
        
        logdic = {f'train_mean_{k}':np.nanmean(list(v.values())) for k,v in md.items() if v}
        mlflow.log_metrics(logdic, step=epoch)
        if wandb.run: wandb.log({k.replace('_','/'):v for k,v in logdic.items()}, step=epoch)
        #if verbose: logger.info(logdic)

        # validation loss and metrics
        with torch.no_grad():

            model.eval()

            loss_sum = 0.
            preactivations_l = []
            activity_idx_l = []
            for nb, batch_indices in enumerate(valid_batcher):

                # get and unpack batch data
                batch_data = Subset(biobert, indices=valid_idx)[batch_indices]
                activity_idx, compound_features, assay_features, _, activity = batch_data

                # move data to device
                # assignment is not necessary for modules but it is for tensors
                # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
                if isinstance(compound_features, torch.Tensor):
                    compound_features = compound_features.to(device)
                assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
                activity = activity.to(device)

                # forward
                if 'Multitask' in hparams['model']:
                    assay_features_norm = F.normalize(
                        assay_features,
                        p=2, dim=1
                    )
                    sim_to_train = assay_features_norm @ train_assay_features_norm.T
                    sim_to_train_weights = F.softmax(
                        sim_to_train * hparams['multitask_temperature'],
                        dim=1
                    )
                    preactivations = model(compound_features, sim_to_train_weights)
                            # forward
                elif hparams.get('loss_fun') in ('CE', 'Con'):
                    preactivations = model.forward_dense(compound_features, 
                        assay_onehot if 'Multitask' in hparams['model'] else assay_features)
                else:
                    preactivations = model(compound_features, assay_features)

                # loss
                preactivations = preactivations*1/hparams.get('beta',1)
                #if hparams.get('loss_fun') in ('CE', 'Con'):
                #    loss = F.binary_cross_entropy_with_logits(preactivations, activity)
                #else:
                loss = criterion(preactivations, activity)

                # accumulate loss
                loss_sum += loss.item()

                if hparams.get('loss_fun') in ('CE','Con'):
                    from mhnreact.utils import top_k_accuracy
                    ks = [1,5,10,50]
                    tkaccs, arocc = top_k_accuracy(torch.arange(0,len(preactivations)), preactivations, k=[1,5,10,50], ret_arocc=True)
                    topk_l.append(tkaccs) #allready detached numpy ;)
                    arocc_l.append(arocc)

                # accumulate preactivations
                # - need to detach; preactivations.requires_grad is True
                # - move it to cpu
                if hparams.get('loss_fun') in ('CE','Con'):
                    #preactivations = preactivations.sum(axis=1)
                    preactivations = torch.diag(preactivations)

                preactivations_l.append(preactivations.detach().cpu())

                # accumulate indices just to double check
                # - activity_idx is a np.array, not a torch.tensor
                activity_idx_l.append(activity_idx)

                if nb % EVERY == 0 and verbose:
                    logger.info(f'Epoch {epoch}: Validation batch {nb} out of {len(valid_batcher) - 1}.')

            # log mean loss over all minibatches
            valid_loss = loss_sum / len(valid_batcher)
            mlflow.log_metric('valid_loss', valid_loss, step=epoch)
            if wandb.run: wandb.log({'valid/loss':valid_loss}, step=epoch)

            # compute test auroc and avgp for each assay (on the cpu)
            preactivations = torch.cat(preactivations_l, dim=0)
            probabilities = torch.sigmoid(preactivations).numpy()

            activity_idx = np.concatenate(activity_idx_l, axis=0)
            #assert np.array_equal(activity_idx, valid_idx)

            targets = sparse.csc_matrix(
                (
                    biobert.activity.data[valid_idx],
                    (
                        biobert.activity.row[valid_idx],
                        biobert.activity.col[valid_idx]
                    )
                ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.bool
            )

            scores = sparse.csc_matrix(
                (
                    probabilities,
                    (
                        biobert.activity.row[valid_idx],
                        biobert.activity.col[valid_idx]
                    )
                ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.float32
            )

            

            md = metrics.swipe_threshold_sparse(
                targets=targets,
                scores=scores, verbose=verbose>=2, ret_dict=True,
            )

            if hparams.get('loss_fun')=='Con':
                for ii, k in enumerate(ks):
                    md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()} #drop last (might be not full)
                md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} #drop last (might be not full)

            # log metrics mean over assays
            
            logdic = {f'valid_mean_{k}':np.nanmean(list(v.values())) for k,v in md.items() if v}
            logdic['valid_loss'] = valid_loss
            
            mlflow.log_metrics(logdic, step=epoch)

            if wandb.run: wandb.log({k.replace('_','/'):v for k,v in logdic.items()}, step=epoch)
            #if verbose: logger.info(logdic)

            # monitor AU-ROC
            if 'valid_mean_davgp' not in logdic:
                logger.info('Using -valid_loss because valid_mean_avgp not in logdic')
            log_value = logdic.get('valid_mean_davgp',-valid_loss)
            #metric_monitor(logdic['valid_mean_davgp'], epoch)
            metric_monitor(log_value, epoch)

            # log model checkpoint dir
            ckpt_dir = loghelper.checkpoint_file
            if wandb.run:
                wandb.run.config.update({'model_save_dir':ckpt_dir})

            if metric_monitor.improvement:
                logger.info(f'Epoch {epoch}: Save model and optimizer checkpoint with val-davgp: {log_value}.')
                torch.save({
                    'value': log_value,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, loghelper.checkpoint_file)

            if metric_monitor.restore:
                # do not load last checkpoint's `optimizer` in order not to
                # loose learning rate changes that may have occurred since then
                logger.info(f'Epoch {epoch}: Restore model (but not optimizer) from checkpoint.')
                checkpoint = torch.load(loghelper.checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict'])

                # update learning rate
                scheduler.step()
                new_lr = [pg['lr'] for pg in optimizer.param_groups]
                if len(new_lr) > 1:
                    logger.warning('There is more than one learning rate. All were updated but only the first one in \'optimizer.param_groups\' is logged.')
                logger.info(f'Epoch {epoch}: Reduce learning rate to {new_lr[0]:.4e}.')
                mlflow.log_metric('lr', new_lr[0], step=epoch)
                if wandb.run: wandb.log({'lr':new_lr[0]},step=epoch)

            if metric_monitor.earlystop:
                logger.info(f'Epoch {epoch}: Out of patience. Early stop!')
                break

            model.train()

        epoch += 1

    # test with best model
    with torch.no_grad():

        epoch -= 1
        logger.info(f'Epoch {epoch}: Restore model from checkpoint.')
        # check if checkpoint exists
        if not os.path.exists(loghelper.checkpoint_file):
            logger.warning(f'Checkpoint file {loghelper.checkpoint_file} does not exist. Test with init model.')
        else:
            checkpoint = torch.load(loghelper.checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        loss_sum = 0.
        preactivations_l = []
        activity_idx_l = []
        for nb, batch_indices in enumerate(test_batcher):

            # get and unpack batch data
            batch_data = Subset(biobert, indices=test_idx)[batch_indices]
            activity_idx, compound_features, assay_features, _, activity = batch_data

            # move data to device
            # assignment is not necessary for modules but it is for tensors
            # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
            activity = activity.to(device)

            # forward
            if 'Multitask' in hparams['model']:
                assay_features_norm = F.normalize(
                    assay_features,
                    p=2, dim=1
                )
                sim_to_train = assay_features_norm @ train_assay_features_norm.T
                sim_to_train_weights = F.softmax(
                    sim_to_train * hparams['multitask_temperature'],
                    dim=1
                )
                preactivations = model(compound_features, sim_to_train_weights)
            else:
                preactivations = model(compound_features, assay_features)

            # loss
            #if hparams.get('loss_fun') in ['CE','Con']:
            #    loss = F.binary_cross_entropy_with_logits(preactivations, activity)
            #else:
            loss = criterion(preactivations, activity)

            # accumulate loss
            loss_sum += loss.item()

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad is True
            # - move it to cpu
            preactivations_l.append(preactivations)

            # accumulate indices just to double check
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx)

            if nb % EVERY == 0 and verbose:
                logger.info(f'Epoch {epoch}: Test batch {nb} out of {len(test_batcher) - 1}.')

        # log mean loss over all minibatches
        mlflow.log_metric('test_loss', loss_sum / len(test_batcher), step=epoch)
        if wandb.run: wandb.log({'test/loss':loss_sum / len(test_batcher)})

        # compute test auroc and avgp for each assay (on the cpu) 'WHY???
        preactivations = torch.cat(preactivations_l, dim=0)
        probabilities = torch.sigmoid(preactivations)

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        #assert np.array_equal(activity_idx, test_idx) #TODO WHY????

        probabilities = probabilities.detach().cpu().numpy().astype(np.float32)

        targets = sparse.csc_matrix(
            (
                biobert.activity.data[test_idx],
                (
                    biobert.activity.row[test_idx],
                    biobert.activity.col[test_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.bool
        )

        scores = sparse.csc_matrix(
            (
                probabilities,
                (
                    biobert.activity.row[test_idx],
                    biobert.activity.col[test_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.float32
        )

        md = metrics.swipe_threshold_sparse(
            targets=targets,
            scores=scores, verbose=verbose>=2, ret_dict=True
        )
        if hparams.get('loss_fun')=='Con':
            for ii, k in enumerate(ks):
                md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()} #drop last (might be not full)
            md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} #drop last (might be not full)

        # log metrics mean over assays
        
        logdic = {f'test_mean_{k}':np.nanmean(list(v.values())) for k,v in md.items() if v}
        mlflow.log_metrics(logdic, step=epoch)

        if wandb.run: wandb.log({k.replace('_','/'):v for k,v in logdic.items()}, step=epoch)
        if verbose:
            logger.info( pd.DataFrame.from_dict([logdic]).T )

        # compute test activity counts and positives
        counts, positives = {}, {}  
        for idx, col in enumerate(targets.T):
            if col.nnz == 0:
                continue
            counts[idx] = col.nnz
            positives[idx] = col.sum()

        #'test_mean_bedroc': 0.6988015835969245, 'test_mean_davgp': 0.16930837444561778, 'test_mean_dneg_avgp': 0.17522445272085613, 'test/mean/auroc': 0.6709850363704437, 'test/mean/avgp': 0.6411171492554743, 'test/mean/neg/avgp': 0.7034156779109996, 'test/mean/argmax/j': 0.4308185
        # store test metrics and counts in a parquet file
        metrics_df = pd.DataFrame(md)
        metrics_df['argmax_j'] = metrics_df['argmax_j'].apply(sigmoid)
        #metrics_df['counts'] = counts #for PC_large: ValueError: Length of values (3933) does not match length of index (615)
        #metrics_df['positives'] = positives

        metrics_df.index.rename('assay_idx', inplace=True)

        metrics_df = biobert.assay_names.merge(metrics_df, left_index=True, right_index=True)
        logger.info(f'Writing test metrics to {loghelper.metrics_file}')
        metrics_df.to_parquet(
            loghelper.metrics_file,
            compression=None,
            index=True
        )

        with pd.option_context('float_format',"{:.2f}".format): 
            print(metrics_df)
            print(metrics_df.mean(0))

        model.train()

    loghelper.stop(keep)



def test(
        biobert: dataset.InMemoryClamp,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        hparams: dict,
        run_info: mlflow.entities.RunInfo,
        device: str = 'cpu',
        verbose: bool = False,
        model = None,
) -> None:
    """
    Test a model on `biobert[test_idx]` if test metrics are not yet to be found
    under the `artifacts` directory. If so, interrupt the program.

    Parameters
    ----------
    biobert: :class:`~biobert.dataset.ImBioBert`
        Dataset instance.
    train_idx: :class:`numpy.ndarray`
        Activity indices of the training split. Only for multitask models.
    test_idx: :class:`numpy.ndarray`
        Activity indices of the test split.
    run_info: :class:`mlflow.entities.RunInfo`
        MLflow's run details (for logging purposes).
    device: str
        Computing device.
    verbose: bool
        Be verbose if True.
    """

    if verbose:
        logger.info('Start evaluation.')

    # set up call-specific logging (run)
    loghelper = LogHelper(run_info.experiment_id, run_info.run_id, logger_format=LOGGER_FORMAT)
    loghelper.start()

    # check that test metrics are not there yet
    assert not loghelper.metrics_file.is_file(), 'Test metrics are already available for this model.'

    # initialize checkpoint
    if model!=None:
        checkpoint = init_checkpoint(loghelper.checkpoint_file, device)
        assert checkpoint, 'No checkpoint found.'
        assert 'model_state_dict' in checkpoint, 'No model found in checkpoint.'

    # initialize model
    if 'Multitask' in hparams['model']:

        _, train_assays = biobert.get_unique_names(train_idx)
        biobert.setup_assay_onehot(size=train_assays.index.max() + 1)
        train_assay_features = biobert.assay_features[:train_assays.index.max() + 1]
        train_assay_features_norm = F.normalize(
            torch.from_numpy(train_assay_features),
            p=2, dim=1
        ).to(device)

        if model!=None:
            model = init_model(
                compound_features_size=biobert.compound_features_size,
                assay_features_size=biobert.assay_onehot.size,
                hp=hparams,
                verbose=verbose
            )

    else:
        if model!=None:
            model = init_model(
                compound_features_size=biobert.compound_features_size,
                assay_features_size=biobert.assay_features_size,
                hp=hparams,
                verbose=verbose
            )

    if verbose:
        logger.info('Load model from checkpoint.')
    if model!=None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # assignment is not necessary when moving modules, but it is for tensors
    # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
    # here I only assign for consistency
    model = model.to(device)

    # initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    test_sampler = SequentialSampler(data_source=test_idx)
    test_batcher = BatchSampler(
        sampler=test_sampler,
        batch_size=hparams['batch_size'],
        drop_last=False
    )

    epoch = checkpoint.get('epoch', 0)
    with torch.no_grad():

        model.eval()

        loss_sum = 0.
        preactivations_l = []
        activity_idx_l = []
        for nb, batch_indices in enumerate(test_batcher):

            # get and unpack batch data
            batch_data = Subset(biobert, indices=test_idx)[batch_indices]
            activity_idx, compound_features, assay_features, assay_onehot, activity = batch_data

            # move data to device
            # assignment is not necessary for modules but it is for tensors
            # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
            activity = activity.to(device)

            # forward
            if 'Multitask' in hparams['model']:
                assay_features_norm = F.normalize(
                    assay_features,
                    p=2, dim=1
                )
                sim_to_train = assay_features_norm @ train_assay_features_norm.T
                sim_to_train_weights = F.softmax(
                    sim_to_train * hparams['multitask_temperature'],
                    dim=1
                )
                preactivations = model(compound_features, sim_to_train_weights)
            else:
                preactivations = model(compound_features, assay_features)

            # loss
            loss = criterion(preactivations, activity)

            # accumulate loss
            loss_sum += loss.item()

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad is True
            # - move it to cpu
            preactivations_l.append(preactivations.detach().cpu())

            # accumulate indices just to double check
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx)

            if nb % EVERY == 0 and verbose:
                logger.info(f'Epoch {epoch}: Test batch {nb} out of {len(test_batcher) - 1}.')

        # log mean loss over all minibatches
        mlflow.log_metric('test_loss', loss_sum / len(test_batcher), step=epoch)
        if wandb.run: wandb.log({'test/loss':loss_sum/len(test_batcher)}, step=epoch)

        # compute test auroc and avgp for each assay (on the cpu)
        preactivations = torch.cat(preactivations_l, dim=0)
        probabilities = torch.sigmoid(preactivations).numpy()

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        assert np.array_equal(activity_idx, test_idx)

        targets = sparse.csc_matrix(
            (
                biobert.activity.data[test_idx],
                (
                    biobert.activity.row[test_idx],
                    biobert.activity.col[test_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.bool
        )

        scores = sparse.csc_matrix(
            (
                probabilities,
                (
                    biobert.activity.row[test_idx],
                    biobert.activity.col[test_idx]
                )
            ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.float32
        )

        md = metrics.swipe_threshold_sparse(
            targets=targets,
            scores=scores, verbose=verbose>=2, ret_dict=True
        )

        # log metrics mean over assays
        logdic = {f'test_mean_{mdk}': np.mean(list(md[f'{mdk}'].values())) for mdk in md.keys()}
        mlflow.log_metrics(logdic, step=epoch)

        if wandb.run: wandb.log({k.replace('_','/'):v for k,v in logdic.items()}, step=epoch)
        if verbose: logger.info(logdic)

        # compute test activity counts and positives
        counts, positives = {}, {}
        for idx, col in enumerate(targets.T):
            if col.nnz == 0:
                continue
            counts[idx] = col.nnz
            positives[idx] = col.sum()

        # store test metrics and counts in a parquet file
        metrics_df = pd.DataFrame(md)
        metrics_df['argmax_j'] = metrics_df['argmax_j'].apply(sigmoid)
        metrics_df['counts'] = counts
        metrics_df['positives'] = positives

        metrics_df.index.rename('assay_idx', inplace=True)

        metrics_df = biobert.assay_names.merge(metrics_df, left_index=True, right_index=True)
        logger.info(f'Writing test metrics to {loghelper.metrics_file}')
        metrics_df.to_parquet(
            loghelper.metrics_file,
            compression=None,
            index=True
        )

        if wandb.run:
            wandb.log({"metrics_per_assay": wandb.Table(data=metrics_df)})

        logger.info(f'Saved best test-metrics to {loghelper.metrics_file}')
        logger.info(f'Saved best checkpoint to {loghelper.checkpoint_file}')

        model.train()

        with pd.option_context('float_format',"{:.2f}".format): 
            print(metrics_df)

        return metrics_df


def get_random_split(activity_df, val_size=0.0, test_size=0.3, seed=0):
    """get random train/val/test split indices in given fraction for each assay"""
    test_idx = activity_df.groupby('assay_idx').sample(frac=test_size, random_state=seed).index.values
    # val_fraction is the fraction of the remaining data
    valid_idx = activity_df.loc[~activity_df.index.isin(test_idx)].groupby('assay_idx').sample(frac=val_size, random_state=seed).index.values
    train_idx = activity_df.loc[~activity_df.index.isin(test_idx) & ~activity_df.index.isin(valid_idx)].index.values
    return train_idx, valid_idx, test_idx

def get_scaffold_split(scaffold_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    # from https://github.com/chainer/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    rng = numpy.random.RandomState(seed)
    
    scaffolds = defaultdict(list)
    for ind, sc in enumerate(scaffold_list):
        scaffolds[sc].append(ind)
    
    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(scaffold_list)))
    n_total_test = int(np.floor(frac_test * len(scaffold_list)))

    train_index = []
    valid_index = []
    test_index = []

    for scaffold_set in scaffold_sets:
        if len(valid_index) + len(scaffold_set) <= n_total_valid:
            valid_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)

    return np.array(train_index), np.array(valid_index), np.array(test_index)   

def get_fewshot_split(activity_df, support_set_size=16, val_size=0.0, seed=0, balanced=False):
    """get few-shot train/valid/test split indices for each assay
        for each assay, support_set_size compounds are sampled and used as trianing set
        val_size is standard 0 and describes the fraction per assay that is used as validation set #TODO change to val_set_size
        balanced: if True, the support set is balanced with equal number of active and inactive compounds per assay
        test_set is the remaining compounds per assay
    """

    gb = ['assay_idx']
    if balanced: 
        gb.append('activity')
        support_set_size = support_set_size // 2
    valid_idx = activity_df.groupby(gb).sample(frac=val_size, random_state=seed).index.values
    train_idx = activity_df.loc[~activity_df.index.isin(valid_idx)].groupby(gb).sample(n=support_set_size, random_state=seed).index.values
    test_idx = activity_df.loc[~activity_df.index.isin(valid_idx) & ~activity_df.index.isin(train_idx)].index.values
    return train_idx, valid_idx, test_idx

def get_dense_split(activity_df, min_cmp_vals=500, min_ass_vals=100):
    cmp_vals = activity_df.groupby('compound_idx').apply(len).sort_values()
    ass_vals = activity_df.groupby('assay_idx').apply(len).sort_values()

    sel_those_cmp = cmp_vals[ (cmp_vals>=min_cmp_vals) ].index #500 - 10% left
    sel_those_ass = ass_vals[ (ass_vals>=min_ass_vals) ].index #100 - 20% left

    sel = activity_df[activity_df.compound_idx.apply(lambda k: k in sel_those_cmp) * activity_df.assay_idx.apply(lambda k: k in sel_those_ass)]

    print(f'sparsity of matrix: {len(sel) / ( len( sel_those_cmp ) * len(sel_those_ass ))*100:1.3f}%', end='\t')
    print(f'#measuremnts: {len(sel)/10e6:3.2f}M, #compounds: {len(sel_those_cmp)/1000:1.1f}k, #assays: {len(sel_those_ass)/1000:1.1f}k')
    
    train_ass = set(sel_those_ass.sort_values()[:int(0.8*len(sel_those_ass))])
    valid_ass = set(sel_those_ass.sort_values()[int(0.8*len(sel_those_ass)):int(0.9*len(sel_those_ass))])
    test_ass = set(sel_those_ass.sort_values()[int(0.9*len(sel_those_ass)):])
    sel['split'] = sel.assay_idx.apply(lambda k: 'train' if k in train_ass else 'valid' if k in valid_ass else 'test')
    train_idx = sel[sel['split']=='train'].index
    valid_idx = sel[sel['split']=='valid'].index
    test_idx = sel[sel['split']=='test'].index
    return train_idx, valid_idx, test_idx

def random(
        biobert: dataset.InMemoryClamp,
        test_idx: np.ndarray,
        run_info: mlflow.entities.RunInfo,
        seed: Optional[int] = 0,
        verbose: bool = False
) -> None:
    """
    Report classification metrics on `biobert[test_idx]` for predictions
    drawn randomly from :math:`\\mathcal{U}(0, 1)`.

    Parameters
    ----------
    biobert: :class:`~biobert.dataset.ImBioBert`
        Dataset instance.
    test_idx: :class:`numpy.ndarray`
        Activity indices of the test split.
    run_info: :class:`mlflow.entities.RunInfo`
        MLflow's run details (for logging purposes).
    seed: int or None
        Set the pseudo-random behavior, if not None.
    verbose: bool
        Be verbose if True.
    """

    if verbose:
        logger.info('Evaluate predictions drawn randomly from U(0, 1).')

    # set up call-specific logging (run)
    loghelper = LogHelper(run_info.experiment_id, run_info.run_id, logger_format=LOGGER_FORMAT)
    loghelper.start()

    rng = default_rng()
    probabilities = rng.uniform(size=test_idx.shape)

    targets = sparse.csc_matrix(
        (
            biobert.activity.data[test_idx],
            (
                biobert.activity.row[test_idx],
                biobert.activity.col[test_idx]
            )
        ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.bool
    )

    scores = sparse.csc_matrix(
        (
            probabilities,
            (
                biobert.activity.row[test_idx],
                biobert.activity.col[test_idx]
            )
        ), shape=(biobert.num_compounds, biobert.num_assays), dtype=np.float32
    )

    md = metrics.swipe_threshold_sparse(
        targets=targets,
        scores=scores, verbose=verbose>=2, ret_dict=True
    )

    # log metrics mean over assays
    logdic = {f'test_mean_{mdk}': np.mean(list(md[f'{mdk}'].values())) for mdk in md.keys()}
    mlflow.log_metrics(logdic, step=0)

    if wandb.run: wandb.log({k.replace('_','/'):v for k,v in logdic.items()}, step=epoch)
    # if verbose: logger.info(logdic)

    # compute test activity counts and positives
    counts, positives = {}, {}
    for idx, col in enumerate(targets.T):
        if col.nnz == 0:
            continue
        counts[idx] = col.nnz
        positives[idx] = col.sum()

    # store test metrics and counts in a parquet file
    metrics_df = pd.DataFrame(
        [
            {k: sigmoid(v) for k, v in md['argmax_j'].items()},  # probabilities instead of scores
            md['auroc'], md['avgp'], md['neg_avgp'], counts, positives
        ],
        index=['argmax_j', 'auroc', 'avgp', 'neg_avgp', 'counts', 'positives']
    ).transpose()
    metrics_df.index.rename('assay_idx', inplace=True)

    metrics_df = biobert.assay_names.merge(metrics_df, left_index=True, right_index=True)
    logger.info(f'Writing test metrics to {loghelper.metrics_file}')
    metrics_df.to_parquet(
        loghelper.metrics_file,
        compression=None,
        index=True
    )


def get_hparams(path, mode, verbose=False):
    """
    Get hyperparameters from a path.
    Parmeters
    ---------
    path: str
        Path to the hyperparameters file.
    mode: str
        Mode of the hyperparameters file. Default is 'logs'.
    verbose: bool
        Be verbose if True.
    """
    if isinstance(path, str):
        path = Path(path)
    hparams = {}
    for fn in os.listdir(path/'params'): 
        try:
            with open(path/f'params/{fn}') as f:
                lines = f.readlines()
                try:
                    hparams[fn] = NAME2FORMATTER.get(fn, str)(lines[0])
                except:
                    hparams[fn] = None if len(lines)==0 else lines[0]
        except:
            pass
    return hparams

def load_model(mlrun_path='',compound_features_size=4096, assay_features_size=2048, device='cuda:0', ret_hparams=False):
    """
    Load a model from a mlflow run.
    Parameters
    ----------
    mlrun_path: str
        Path to the mlflow run.
    device: str
        Device to load the model on.
    """
    if isinstance(mlrun_path, str):
        mlrun_path = Path(mlrun_path)
    
    hparams = get_hparams(
            path=mlrun_path,
            mode='logs',
            verbose=True
        )


    if compound_features_size is None:
        elp = Path(hparams['dataset']) / ('compound_features_'+hparams['compound_mode']+'.npy')
        try:
            compound_features_size = np.load(elp).shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f'Compound features file {elp} not found. Add as input to this method.')

    if assay_features_size is None:
        elp = Path(hparams['dataset']) / ('assay_features_'+hparams['assay_mode']+'.npy')
        try:
            assay_features_size = np.load(elp).shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f'Assay features file {elp} not found. Add as input to this method.')

    model = init_model(
                    compound_features_size=compound_features_size,#hparams['compound_layer_sizes'][0],
                    assay_features_size=assay_features_size,#hparams['assay_layer_sizes'][0],
                    hp=hparams,
                    verbose=True
                )

    # load in the model and generate hidden
    checkpoint = init_checkpoint(mlrun_path/'artifacts/checkpoint.pt', device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if ret_hparams:
        return model, hparams
    return model

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def bootstrap_metric(function, n=500):
    "wrapper for metrics to bootstrap e.g. calc std"
    def wrapper(y_true, y_pred, sample_weight=None):
        l = (len(y_true))
        res = []
        for i in range(n):
            s = np.random.choice(range(l), l, replace=True)
            if not len(np.unique(y_true[s]))==2:
                continue
            else:
                res.append( function(y_true[s], y_pred[s], sample_weight=None if sample_weight is None else sample_weight[s]))#,
        return np.array(res)
    return wrapper

def get_default_split(dset):
    "Get default split for a dataset. Warning not foolproof. Use with caution."
    dset = dset.lower()
    if 'pubchem' in dset: return 'time_a'
    if 'tox21_10k' in dset: return 'split'
    elif 'tox21_original' in dset: return 'original_split'
    elif 'uspto' in dset: return 'time_split'
    else: return 'scaffold_split'

def set_device(gpu=0, verbose=False):
    "Set device to gpu or cpu."
    if gpu=='any':
        gpu = 0
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    return device

def init_optimizer(model, hp, verbose=False):
    """
    Initialize optimizer.
    """
    if verbose:
        logger.info(f"Trying to init '{hp['optimizer']}' optimizer from torch.optim")
    hp['lr'] = hp.pop('lr_ini')
    hp['weight_decay'] = hp.pop('l2')
    optimizer = getattr(torch.optim, hp['optimizer'])
    filtered_dict = filter_dict(hp, optimizer)
    return optimizer(params=model.parameters(), **filtered_dict) #also accepts a lot of new hps)

def filter_dict(dict_to_filter, thing_with_kwargs):
    """
    filters dict_to_filter by the arguments of the object or function of thing_with_kwargs
    so that you can stressfree do this: thing_with_kwargs(**filter_dict_return)
    returns: filtered_dict
    modified from from 
    https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function
    
    """
    import inspect
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    inters = set(dict_to_filter.keys()).intersection(set(filter_keys))
    return {k:dict_to_filter[k] for k in inters}


def init_checkpoint(path, device, verbose=False):
    """loads from path if path is not None, else returns empty dict"""
    if path is not None:
        if verbose:
            logger.info('Load checkpoint.')
        return torch.load(path, map_location=device)
    return {}