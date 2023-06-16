from clamp.utils import set_device

from clamp.dataset import InMemoryClamp
from clamp import utils

from pathlib import Path
from clamp.train import setup_dataset

import mlflow
import os
import argparse
import torch
import numpy as np
import pandas as pd
from loguru import logger

from clamp.utils import batch, bootstrap_metric
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings
from sklearn.exceptions import ConvergenceWarning

"""
example call:
python clamp/linear_probe.py ./data/moleculenet/bace_c/ --split=scaffold_split --compound_mode=morganc+rdkc --compound_features_size=8192
"""

ACCEPTED_CLFS = ['LR','RF','kNN']

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def compute_features(MNET_NEW_DIR, fptype='morganc+rdkc', fp_size=8192, njobs=25, standardize_mols=False):
    import numpy as np
    from mhnreact.molutils import convert_smiles_to_fp
    import pandas as pd

    DIR = MNET_NEW_DIR+'/'
    compounds = pd.read_parquet(DIR+'compound_smiles.parquet')['CanonicalSMILES'].values.tolist()
    if standardize_mols:
        from clamp.mol_utils import standardize
        print("standardizing the compounds")
        compounds = list(map(standardize, compounds))
    if fptype=='MxFP':
        fptype = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
    return convert_smiles_to_fp(compounds, fp_size=fp_size, which=fptype,radius=2, njobs=njobs).astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser('Test a Clamp-Activity model.')
    parser.add_argument('dataset', help='Path to a prepared dataset directory.')
    parser.add_argument('--split',type=str, default='scaffold_split',help="split-type default: scaffold_split, other options random_{seed}, or column of activity.parquet triplet")
    parser.add_argument('--run_dir', default='LR', help='Path to an MLflow run directory. If None is provided, LogisticRegression is uesd, if RF is provided Random Forst clf is used')
    parser.add_argument('--gpu', type=str, default='any', help='GPU number to use or "any" if any GPU is fine.', metavar='')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs to average over')
    parser.add_argument('--verbose','-v', type=int, default=0, help='verbosity level default=0')
    parser.add_argument('--assay_features_size', type=int, default=None, help='size of assay features, otherwise derived from dataset')
    parser.add_argument('--compound_features_size', type=int, default=None, help='size of assay features, otherwise derived from dataset')
    parser.add_argument('--compound_mode', type=str, default=None, help='fp-type for compound features, otherwise derived from model')
    parser.add_argument('--train_set_size',type=float, default=-1, help="per task how many train samples to use, -1 for all")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--experiment', type=str, default='', help="experiment name")
    parser.add_argument('--wandb_id', type=str, default=None, help="wandb id to log to")
    args = parser.parse_args()
    return args

def davgp_score(y_true, y_pred, sample_weight=None):
    avgp = average_precision_score(y_true, y_pred, sample_weight=sample_weight)
    y_avg = np.average(y_true, weights=sample_weight)
    return avgp - y_avg

def compute_metrics(y_true, y_pred, bootstrap_n=500):
    if not len(np.unique(y_true))==2:
        print('only one class in test set')
        return {'roc_auc':np.nan, 'avgp':np.nan, 'davgp':np.nan, 'roc_auc_std':np.nan, 'avgp_std':np.nan, 'davgp_std':np.nan}
    roc_auc = roc_auc_score(y_true, y_pred)
    roc_auc_std = bootstrap_metric(roc_auc_score, n=bootstrap_n)(y_true, y_pred).std()

    avgp = average_precision_score(y_true, y_pred)
    avgp_std = bootstrap_metric(average_precision_score, n=bootstrap_n)(y_true, y_pred).std()

    davgp = avgp - y_true.mean()
    davgp_std = bootstrap_metric(davgp_score, n=bootstrap_n)(y_true, y_pred).std()

    return {'roc_auc': roc_auc, 'avgp': avgp, 'davgp':davgp, 'roc_auc_std': roc_auc_std, 'avgp_std': avgp_std, 'davgp_std':davgp_std}

def linear_probe(dataset='./data/MoleculeNet/bace_c/', assay_mode='', compound_mode='morganc+rdkc', split='scaffold_split', run_dir=None, gpu='any', 
                n_runs=1, verbose=False, assay_features_size=None, compound_features_size=None, train_set_size=-1, seed=42, standardize_mols=False, **kwargs):
    """
    Run linear probe on a dataset. If no run_dir for the pretrained model is provided, uses LogisticRegression
    returns: dict of results
    """
    if compound_features_size is None: #load from dataset
        #if os.exists(f'{dataset}compound_features_{compound_mode}.npy'):
        clamp, train_idx, valid_idx, test_idx = setup_dataset(dataset=dataset, assay_mode=assay_mode, compound_mode=compound_mode,
            split=split, verbose=verbose, support_set_size=0, drop_molnet_mols=False, drop_molnet_assays=False, drop_cids_path=None, train_only_actives=False, seed=seed, **kwargs)
        X_fps = clamp.compound_features#.to(device)
        logger.info(f'loaded compound features from {dataset}compound_features_{compound_mode}.npy, shape: {X_fps.shape}')
    else:
        print('load with default compound_mode, and compute them later')
        clamp, train_idx, valid_idx, test_idx = setup_dataset(dataset=dataset, assay_mode=assay_mode, compound_mode='morganc+rdkc',
            split=split, verbose=verbose, support_set_size=0, drop_molnet_mols=False, drop_molnet_assays=False, drop_cids_path=None, train_only_actives=False, seed=seed, **kwargs)
        print(f'compute features: {compound_mode}')
        X_fps = compute_features(dataset, fptype=compound_mode, fp_size=compound_features_size, njobs=1, standardize_mols=standardize_mols)

    device = set_device(gpu=gpu, verbose=verbose)

    fp_size = X_fps.shape[1]
    fp_type = compound_mode

    if run_dir is None or run_dir=='LR' or run_dir=='RF' or ('kNN' in run_dir):
        model = None
    else:
        run_dir = Path(run_dir)
        model = utils.load_model(run_dir, 
                                compound_features_size=compound_features_size if compound_features_size else clamp.compound_features.shape[1], 
                                assay_features_size=assay_features_size if assay_features_size else clamp.assay_features_size, 
                                device=device)
        model.eval().to(device)

        X_ce = []
        for bi in batch((range(len(X_fps))), 512):
            tmp = model.compound_encoder(torch.from_numpy(X_fps[bi]).to(device).float()).detach().cpu().numpy() #.toarray() 
            X_ce.append(tmp)
        X_ce = np.concatenate(X_ce) 

    seeds = range(n_runs)
    mean_rocaucs = []
    df_res = pd.DataFrame()
    for seed in seeds:
        roc_aucs = []
        roc_aucs2 = []
        #print(clamp.assay_names)
        for task_i in clamp.activity_df.assay_idx.iloc[train_idx].unique():
            if verbose: print(f'Running task {task_i}...')
            subset = clamp.activity_df.iloc[train_idx]
            train_subset = subset[subset.assay_idx==task_i]
            
            test_subset = clamp.activity_df.loc[test_idx]
            test_subset = test_subset[test_subset.assay_idx==task_i]
            cidx_test = test_subset.compound_idx.values

            if train_set_size>=0:
                if train_set_size<1:
                    train_subset = train_subset.sample(frac=train_set_size, random_state=seed)
                else:
                    train_subset = train_subset.sample(int(train_set_size), random_state=seed)
            
            cidx = train_subset.compound_idx.values
            
            y = train_subset.activity.values
            #clf = SGDClassifier('log')


            clf = LogisticRegression(max_iter=1500, class_weight='balanced', C=1, random_state=seed)
            if run_dir=='RF':
                clf = RandomForestClassifier(n_estimators=701, random_state=seed, class_weight="balanced", )
                if task_i==0:
                    print('training RF model')
            if run_dir=='kNN':
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_neighbors=len(cidx), weights='distance', metric='cosine')
                if task_i==0:
                    print('training distance-weighted cosine-kNN model')
            clf.fit(X_fps[cidx], y>0)
            y_test = test_subset.activity.values>0
            y_pred = clf.predict_proba(X_fps[cidx_test])[:,1]

            mtrcs = compute_metrics(y_test, y_pred)
            roc_aucs.append(mtrcs['roc_auc'])
            df_res = df_res.append( {'seed':seed, 'dataset':dataset, 'task_i':task_i, 'clf':f'{"LR" if len(str(run_dir))>=5 else str(run_dir)} {fp_type}-{fp_size} {split}-split (ours)',
                                     'fp_type':fp_type, 'fp_size':fp_size, 'train_set_size':train_set_size, **mtrcs}, ignore_index=True)

            # model for linear probing
            if model is not None:
                clf2 = LogisticRegression(max_iter=1500, class_weight='balanced', C=1, random_state=seed)
                clf2.fit(X_ce[cidx], y>0)
                y_pred2 = clf2.predict_proba(X_ce[cidx_test])[:,1]
                mtrcs2 = compute_metrics(y_test, y_pred2)
                roc_aucs2.append(mtrcs2['roc_auc'])
                
                hparams = utils.get_hparams(
                                path=Path(run_dir),
                                mode='logs',
                                verbose=verbose,
                            )
                # str(type(model)).split('.')[-1][:-2] # get model name
                clfd = f' {hparams["model"]} {fp_type} {hparams["assay_mode"]} {split}-split {run_dir} (ours)'
                df_res = df_res.append( {'seed':seed, 'dataset':dataset, 'task_i':task_i, 'clf':clfd,
                                     'fp_type':fp_type, 'fp_size':fp_size, 'train_set_size':train_set_size, 'assay_mode':hparams["assay_mode"], 
                                     'split':split, 'run_dir':run_dir,
                                     **mtrcs2}, ignore_index=True)
            else:
                roc_auc2 = np.nan


            if verbose>=1:
                print(f'{clamp.assay_names.loc[task_i].values[1]}: {mtrcs["roc_auc"]:.3f}  with comp_enc {roc_auc2:.3f}')
    return df_res

if __name__ == '__main__':
    args = parse_args()

    if args.run_dir is None or args.run_dir in ACCEPTED_CLFS:
        hparams = {'compound_mode':None, 'assay_features_size':None, 'compound_features_size':None}
    else:
        hparams = utils.get_hparams(
                path=Path(args.run_dir),
                mode='logs',
                verbose=args.verbose,
            )

    #print(hparams)

    dataset_args = args.__dict__.copy()
    print(args.compound_mode)
    dataset_args['compound_mode'] = args.compound_mode if args.compound_mode is not None else hparams['compound_mode'] #TODO
    dataset_args['assay_mode'] = ''#hparams['assay_mode']

    print(dataset_args)

    df_res = linear_probe(dataset=args.dataset, assay_mode=dataset_args['assay_mode'], compound_mode=dataset_args['compound_mode'], 
                            split=args.split, run_dir=args.run_dir, gpu=args.gpu, n_runs=args.n_runs, verbose=args.verbose, 
                            assay_features_size=args.assay_features_size, compound_features_size=args.compound_features_size, 
                            train_set_size=args.train_set_size, seed=args.seed)

    df_tmp = df_res.groupby(['clf']).mean()
    print('roc_auc:', df_tmp['roc_auc'], 'davgp:', df_tmp['davgp'])
    out_fn = f'./data/experiments/linear_probing/{args.run_dir.replace("/","").replace(".","")}_{dataset_args["compound_mode"]}_{args.dataset.replace("/","%")}_{args.split}_{args.seed}_linear_probing.csv'
    print('saved to', out_fn)
    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn))
    df_res.to_csv(out_fn)

    if args.wandb_id:
        import wandb
        #write into wandb old run
        wandb.init(project='clamp', entity='phseidl', id=args.wandb_id, resume=True)
        for clm in df_tmp.columns:
            wandb.log({f"linear_probing/{args.dataset}/{args.split}/{clm}": df_tmp[clm].values[0]})
        #wandb.finish()
        
