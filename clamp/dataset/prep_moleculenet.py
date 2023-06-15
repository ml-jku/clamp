import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from collections import defaultdict
from rdkit import Chem
from pathlib import Path
import argparse
import os
import tqdm
import tqdm
import sys

"""
Prepare MoleculeNet datasets for use with CLAMP.
example call:
python clamp/dataset/prep_moleculenet.py --mnet_dir=./data/moleculenet/
"""


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser('Prepare MoleculeNet datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mnet_dir',default='./data/moleculenet/', help='Path to MoleculeNet directory.')

    args = parser.parse_args()
    MNET_NEW_DIR = Path(args.mnet_dir)

    try:
        import deepchem
    except:
        print("DeepChem not installed. Please install DeepChem e.g. ````pip install deepchem``` to use this script.")
        sys.exit(1)

    dsets = ['bace_c','bbbp','clintox','hiv','sider','tox21','toxcast']

    dataset_loading_functions = {
        'bace_c': deepchem.molnet.load_bace_classification,
        'bace_r': deepchem.molnet.load_bace_regression,
        'bbbp': deepchem.molnet.load_bbbp,
        'chembl': deepchem.molnet.load_chembl,
        'clearance': deepchem.molnet.load_clearance,
        'clintox': deepchem.molnet.load_clintox,
        'delaney': deepchem.molnet.load_delaney,
        'factors': deepchem.molnet.load_factors,
        'hiv': deepchem.molnet.load_hiv,
        'hopv': deepchem.molnet.load_hopv,
        'hppb': deepchem.molnet.load_hppb,
        'kaggle': deepchem.molnet.load_kaggle,
        'kinase': deepchem.molnet.load_kinase,
        'lipo': deepchem.molnet.load_lipo,
        'muv': deepchem.molnet.load_muv,
        'nci': deepchem.molnet.load_nci,
        'pcba': deepchem.molnet.load_pcba,
        'ppb': deepchem.molnet.load_ppb,

        'qm8': deepchem.molnet.load_qm8,
        'qm9': deepchem.molnet.load_qm9,
        'sampl': deepchem.molnet.load_sampl,
        'sider': deepchem.molnet.load_sider,
        'thermosol': deepchem.molnet.load_thermosol,
        'tox21': deepchem.molnet.load_tox21,
        'toxcast': deepchem.molnet.load_toxcast,
        'uv': deepchem.molnet.load_uv
    }

    for dset in tqdm.tqdm(dsets, desc='Datasets'):
        print(dset)
        tasks, all_dataset, transformers = dataset_loading_functions[dset](featurizer='Raw', split='scaffold') #8/1/1 split

        X = [[Chem.MolToSmiles(xi) for xi in ii.X] for ii in all_dataset]
        y = all_dataset[0].y #only train

        activity_triplet = []
        compound2index = {} 
        activity2index = {}

        dir = MNET_NEW_DIR/dset
        if not dir.exists():
             os.makedirs(dir)

        for spli, spli_i in zip(['train','valid','test'],range(3)):

            for task_i in tqdm.tqdm(range(y.shape[1]), desc=f'{dset} Tasks split {spli}'):
                for ii, sample in enumerate(X[spli_i]):
                    cidx = compound2index.get(sample, len(compound2index))
                    compound2index[sample] = cidx #overrides or new entry

                    assay_name = dset+' '+tasks[task_i]
                    aidx = activity2index.get(assay_name, len(activity2index))
                    activity2index[assay_name] = aidx #overrides or new entry

                    activity = all_dataset[spli_i].y[ii,task_i]
                    activity_triplet.append({'compound_idx':cidx,'assay_idx':aidx, 'activity':(activity>0)*1, 
                                            'activity_numeric':activity, 'scaffold_split':spli})
        activity_triplet_df = pd.DataFrame(activity_triplet)
        activity_triplet_df
        print(activity_triplet_df.assay_idx.unique())
        print(activity_triplet_df.activity_numeric.unique())
        
        #save 
        # exists_ok=True not possible
        activity_triplet_df.to_parquet(dir/'activity.parquet')
        
        compound2index_df = pd.DataFrame()
        compound2index_df['CID'] = compound2index.values()
        compound2index_df['CanonicalSMILES'] = compound2index.keys()
        compound2index_df
        
        compound2index_df[['CID']].to_parquet(dir/'compound_names.parquet')
        compound2index_df.to_parquet(dir/'compound_smiles.parquet')
        
        assay2index_df = pd.DataFrame()
        assay2index_df['AID'] = activity2index.values()
        assay2index_df['description'] = activity2index.keys()
        assay2index_df.to_parquet(dir/'assay_names.parquet')