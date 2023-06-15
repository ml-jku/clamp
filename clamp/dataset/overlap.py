from mhnreact.molutils import disable_rdkit_logging
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from clamp.train import setup_dataset
from clamp.mol_utils import standardize
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm.contrib.concurrent import process_map
tqdm.pandas()
disable_rdkit_logging()

"""check for overlap between two datasets
    dset1: path to the dataset to check
    downstream_dsets: list of downstream datasets to check
    test_set_only: only check for the test-set molecules of the downstream datasets
    standardize_smiles: standardize the smiles before checking
    returns a dictionary of the downstream datasets and the pubchem cids to drop

example call:
python clamp/dataset/overlap.py \
    --dset_path=./data/pubchem23/ \
    --standardize_smiles=True \
    --cids_overlap_path=./data/pubchem23/cidx_overlap_moleculenet.npy \
        --downstream_dsets ./data/moleculenet/tox21/ ./data/moleculenet/toxcast ./data/moleculenet/bace_c ./data/moleculenet/hiv \
        ./data/moleculenet/bbbp ./data/moleculenet/sider ./data/moleculenet/clintox ./data/moleculenet/tox21_10k 
"""

def smiles2inchii(smiles, sanitize=False):
    mol = None
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if mol is None:
            return smiles
        if sanitize: # only rudamentary sanitization -- use sanitize from mol_utils
            faild_op = Chem.SanitizeMol(mol, catchErrors=True)
            AllChem.FastFindRings(mol) #Providing ring info
        mol.UpdatePropertyCache(strict=False) #Correcting valence info # important operation
        return AllChem.MolToInchi(mol)
    except:
        pass
    return smiles

def compute_inchii(df, save_df=False, save_path=None, standardize_smiles=True, njobs=50):
    scn = 'StandardizedSMILES' if standardize_smiles else 'CanonicalSMILES'
    if standardize_smiles:
        if scn not in df.columns:
            #df[scn] = df.CanonicalSMILES.progress_apply(lambda k: (standardize(str(k))))
            df[scn] = process_map(standardize, df.CanonicalSMILES.apply(str).values.tolist(), max_workers=njobs, chunksize=10, desc='standardizing smiles')
            # save it
            if save_df:
                df.to_parquet(save_path/'compound_smiles.parquet')
        else:
            logger.info(f'{scn} column already exists in the dataset, using that')
    df['inchii'] = process_map(smiles2inchii, df[scn].apply(str).values.tolist(), max_workers=njobs, chunksize=10, desc='converting to inchii')
    return df

def check_overlap(dset1, downstream_dsets, 
         test_set_only=False, standardize_smiles=True, njobs=50) -> dict:
    """
    given a dataset, check if any of the downstream datasets have the same compounds
    dset1: path to the dataset to check
    downstream_dsets: list of downstream datasets to check
    test_set_only: only check for the test-set molecules of the downstream datasets
    standardize_smiles: standardize the smiles before checking
    returns a dictionary of the downstream datasets and the pubchem cidx to drop
    """
    dset1 = Path(dset1)
    
    pubchem_mols_df = pd.read_parquet(dset1/'compound_smiles.parquet')
    pubchem_mols_df = compute_inchii(pubchem_mols_df, save_df=True, save_path=dset1, standardize_smiles=standardize_smiles, njobs=njobs)
    
    dset2drop_pubchem_cidx = {}

    for dset in downstream_dsets:
        dset_path = Path(dset)
        
        split = 'scaffold_split'
        if 'tox21_original' in dset: split='original_split'
        if 'tox21_10k' in dset: split='split'

        biobert, train_idx, valid_idx, test_idx = setup_dataset(dataset=dset_path, assay_mode='', compound_mode='morganc+rdkc',split=split, verbose=False)
        test_cmp_idxs = biobert.activity_df.compound_idx.unique()
        if test_set_only:
            test_cmp_idxs = biobert.activity_df.loc[test_idx].compound_idx.unique()
        
        m2_df = pd.read_parquet(dset_path/'compound_smiles.parquet')
        
        m2_df_ss = compute_inchii(m2_df.loc[test_cmp_idxs], standardize_smiles=standardize_smiles, njobs=njobs)

        inchii_set = set(m2_df_ss.inchii)
        # get the idx of those in pubchem
        drop_pubchem_idxs = pubchem_mols_df.inchii.apply(lambda k: k in inchii_set)
        drop_perc = drop_pubchem_idxs.mean()
        dset2drop_pubchem_cidx[dset] = drop_pubchem_idxs[drop_pubchem_idxs].index
        logger.info(f'{dset} found {len(dset2drop_pubchem_cidx[dset])} overlapping compounds from {dset}, dropping {drop_perc*100:.2f}% of compounds')
        # cid pubchem_mols_df[drop_pubchem_idxs].CID.unique()
        
        print(dset, len(dset2drop_pubchem_cidx[dset]), 'from', len(test_cmp_idxs))

    return dset2drop_pubchem_cidx

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_path', type=str, default='./data/pubchem23/')
    parser.add_argument('--cidx_overlap_path', type=str, default='./data/pubchem23/cidx_overlap.npy', help='path to save the cidx to drop from pubchem')
    # list of strings
    downstream_dsets = ['./data/moleculenet/tox21/',
                    './data/moleculenet/toxcast','./data/moleculenet/bace_c','./data/moleculenet/hiv',
                    './data/moleculenet/bbbp','./data/moleculenet/sider','./data/moleculenet/clintox','./data/moleculenet/tox21_10k']
    parser.add_argument('--downstream_dsets',type=str, nargs='+', default=downstream_dsets, 
                        help='list of strings of downstream datasets default see source code')
    parser.add_argument('--split_col', type=str, default='scaffold_split')
    parser.add_argument('--test_set_only', type=bool, default=False)
    parser.add_argument('--njobs', help='njobs', default=50, type=int)
    parser.add_argument('--standardize_smiles', type=bool, default=True)
    
    args = parser.parse_args()

    dset2drop_pubchem_cidx = check_overlap(dset1=args.dset_path, 
    downstream_dsets=downstream_dsets, test_set_only=args.test_set_only, standardize_smiles=args.standardize_smiles, njobs=args.njobs)

    # save dset2drop_pubchem_cidx to json file
    #import json
    #with open(args.cidx_overlap_path, 'w') as fp:
    #    json.dump(dset2drop_pubchem_cidx, fp)


    all_idxs = set()
    [all_idxs.update(list(v)) for v in dset2drop_pubchem_cidx.values()]

    logger.info(f'found {len(all_idxs)} pubchem cidx to drop')
    logger.info(f'saving to {args.cidx_overlap_path}')

    
    np.save(args.cidx_overlap_path, np.array(all_idxs))