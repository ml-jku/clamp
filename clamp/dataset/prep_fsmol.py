import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
import argparse
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import os
import sys
import multiprocessing
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from chembl_webresource_client.new_client import new_client
assay_chembl_client = new_client.assay

"""
Prepare FS-Mol datasets for use with CLAMP.
example call:
python clamp/dataset/prep_fsmol.py --data_dir=./data/fsmol/
"""

def download_fsmol(dir: str):
    """Download the FS-Mol dataset.
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    # current dir
    curr_dir = os.getcwd()
    os.chdir(dir)
    URL = 'https://figshare.com/ndownloader/files/31345321' # fsmol.tar
    # download and name fsmol.tar
    if not os.path.exists('./fs-mol/'):
        logger.info('Extracting fsmol.tar')
        os.system('wget -O fsmol.tar ' + URL)
        os.system('tar -xvf fsmol.tar')
        # delete fsmol.tar
        os.system('rm fsmol.tar')
    else:
        logger.info('FS-Mol already extracted')
    os.system('git clone https://github.com/microsoft/FS-Mol')
    os.chdir(curr_dir)


def _prepro_fsmol(dir: str, dset_version='fsmol-0.1'):
    """Preprocess the FS-Mol dataset.
    dset_version: fsmol-0.1
        for future versions of the dataset,
        currently only one version is available
        if None is provided it will use the entire dataset
    """
    sys.path.insert(0, dir)
    sys.path.insert(0, os.path.join(dir, 'FS-Mol/'))
    from fs_mol.data import FSMolDataset, DataFold

    logger.info(f'loading fsmol version {dset_version}')
    dataset = FSMolDataset.from_directory(os.path.join(dir,'fs-mol/'), 
            task_list_file=os.path.join(dir,f'FS-Mol/datasets/{dset_version}.json') if dset_version else None)
    logger.info('load fsmol')
    task_iterable_test = dataset.get_task_reading_iterable(DataFold.TEST)
    task_iterable_train = dataset.get_task_reading_iterable(DataFold.TRAIN)
    task_iterable_valid = dataset.get_task_reading_iterable(DataFold.VALIDATION)

    #a = next(iter(task_iterable_test))
    # test tasks in train tasks
    
    tasks_dic = defaultdict(list)
    activity_triplet = []
    compound2index = {} 
    activity2index = {}

    def add_or_insert(some_dict, entry):
        nid = some_dict.get(entry, len(some_dict))
        some_dict[entry] = nid #overrides or new entry
        return some_dict, nid

    logger.info('preprocessing fsmol - convert to CLAMP format')
    for spli, task_iter in zip(['train','valid','test'],[task_iterable_train, task_iterable_valid, task_iterable_test]):
        print(spli)
        for task in tqdm(iter(task_iter)):
            tasks_dic[spli].append(task.name)
            for sample in task.samples:
                #compound2index, cidx = add_or_insert(compound2index, sample.smiles) #should do the same thing ;)
                cidx = compound2index.get(sample.smiles, len(compound2index))
                compound2index[sample.smiles] = cidx #overrides or new entry
                
                aidx = activity2index.get(sample.task_name, len(activity2index))
                activity2index[sample.task_name] = aidx #overrides or new entry
                
                activity_triplet.append({'compound_idx':cidx,'assay_idx':aidx, 'activity':sample.bool_label*1, 
                                        'activity_numeric':sample.numeric_label, 'FSMOL_split':spli, })

    dir = Path(dir)
    activity_triplet_df = pd.DataFrame(activity_triplet)
    logger.info('saving fsmol activity-triplet df to activity.parquet')
    activity_triplet_df.to_parquet(dir/'activity.parquet')
    compound2index_df = pd.DataFrame()
    compound2index_df['CID'] = compound2index.values()
    compound2index_df['CanonicalSMILES'] = compound2index.keys()
    compound2index_df[['CID']].to_parquet(dir/'compound_names.parquet')

    compound2index_df.to_parquet(dir/'compound_smiles.parquet')

    assay2index_df = pd.DataFrame()
    assay2index_df['AID'] = activity2index.values()
    assay2index_df['CHEMBL_ID'] = activity2index.keys()

    logger.info('getting assay descriptions from chembl')
    ress = get_chembl_description(list(activity2index.keys()))

    df_assay = pd.DataFrame(ress).T
    td = {k:set(v) for k,v in tasks_dic.items()}

    for split in td.keys():
        df_assay.loc[td[split],'split'] = split

    assay2index_df_merge = assay2index_df.join(df_assay, on='CHEMBL_ID', how='left')
    assay2index_df_merge
    logger.info('saving fsmol assay-description df to assay_names.parquet')
    assay2index_df_merge.to_parquet(dir/'assay_names.parquet')

    assert assay2index_df_merge['description'].isna().sum() == 0, 'some assays have no description'
    assert assay2index_df_merge['split'].isna().sum() == 0, 'some assays have no split'
    if dset_version=='fsmol-0.1':
        assert len(assay2index_df_merge) == 5135, 'number of assays should be 5135 for fsmol-0.1'
    

def get_chembl_description(list_of_chembl_ids):
    ress = {}
    for assay_chembl_id in tqdm(list_of_chembl_ids):
        try:
            res = assay_chembl_client.filter(assay_chembl_id=assay_chembl_id)
            ress[assay_chembl_id] = res[0]
        except:
            ress[assay_chembl_id] = {}
            print('error with', assay_chembl_id)
        # len(res) produces error check if res[1] exists
        #try:
        #    res[1]
        #    logger.info(f"multiple results for {assay_chembl_id} using the first one")
        #except IndexError:
        #    pass
    return ress

if __name__ == '__main__':
    # parese arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/fsmol/')
    # cores
    parser.add_argument('--cores', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--skip_download', action='store_true', default=False)
    args = parser.parse_args()

    if args.skip_download:
        print('Skip downloading PubChem dataset')
    else:
        download_fsmol(args.data_dir)

    _prepro_fsmol(args.data_dir, dset_version=None)

