import os
from typing import Union
from pathlib import Path
import gzip
import xml.etree.ElementTree as ET
import numpy as np
import os
import zipfile
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from joblib import Parallel, delayed
import multiprocessing
import argparse
from loguru import logger

"""
example call:
python clamp/dataset/prep_pubchem.py --data_dir=./data/pubchem23/
"""

def download_pubchem(dir: str):
    """ Download or update the PubChem dataset """
    cur_dir = os.getcwd()
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.chdir(dir)
    # output result from cmd
    # update instead of re-downloading
    os.system('wget -N -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/')
    os.system('wget -N -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Description/')
    os.system('wget -N -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/AssayNeighbors/')
    #os.system('wget -N -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/')#
    os.system('wget -N -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz')
    print('Downloaded PubChem dataset to', dir)
    # delete pubchem18.zip
    os.system('rm -rf ftp.ncbi.nlm.nih.gov')
    # jump back to original dir
    os.chdir(cur_dir)

def _prepro_bioassay(dir: str, ret_all=False):
    """Preprocess the PubChem Bioassay dataset.
    dir:
        path to the bioassay csv file
    ret_all:
        if True, return the df with first two rows containing column descriptions
    returns:
        df with columns: cid, activity, activity_numeric
    """
    df = pd.read_csv(dir, compression='gzip', sep=',', quotechar='"', usecols=[2,3,4,5])#, engine="pyarrow")
    if ret_all:
        return df
    df = df.iloc[2:] # first 2 rows are index

    #rename activity and cid columns
    df = df.rename(columns={'PUBCHEM_CID':'cid', 'PUBCHEM_ACTIVITY_SCORE': 'activity_numeric'})
    # map 'Active' to 1 and 'Inactive' to 0
    df['PUBCHEM_ACTIVITY_OUTCOME'] = df['PUBCHEM_ACTIVITY_OUTCOME'].astype('category')
    #ValueError: cannot set a frame with no defined index and a scala
    df['activity'] = np.nan
    # if there are actives and inactives
    if 'Active' in df.PUBCHEM_ACTIVITY_OUTCOME.cat.categories:
        df.loc[df.PUBCHEM_ACTIVITY_OUTCOME=='Active', 'activity'] = 1
    if 'Inactive' in df.PUBCHEM_ACTIVITY_OUTCOME.cat.categories:
        df.loc[df.PUBCHEM_ACTIVITY_OUTCOME=='Inactive', 'activity'] = 0
    #df['activity'] = df['activity'].replace({'Active': 1, 'Inactive': 0, 'Inconclusive': np.nan}) # Inconclusive is NaN and will be dropped
    # drop rows where cid or activity is Nan
    df = df.dropna(subset=['activity', 'cid'])
    df['cid'] = df['cid'].astype('int')
    df['activity_numeric'] = df['activity_numeric'].astype('float')

    return df[['cid','activity','activity_numeric']]


def prepro_all_bioassay(fn: Union[str, Path], ret_all=False):
    with open(fn, 'rb') as zf:
        z = zipfile.ZipFile(zf)
        all_df = pd.DataFrame()
        for filename in (z.namelist()):
            if filename.endswith('.csv.gz'):      
                with z.open(filename) as f:
                    df = _prepro_bioassay(f, ret_all=ret_all)
                    df['aid'] = filename.split('/')[-1].split('.')[0]
                    all_df = pd.concat([all_df, df], ignore_index=True)
            if filename.endswith('.xml.gz'): # description file
                with z.open(filename) as f:
                    desc = gzip.decompress(f.read())
                    root = ET.fromstring(desc)
                    di = {}
                    for child in root:
                        di[child.tag] = child.text
                        for grandchild in child:
                            di[grandchild.tag] = grandchild.text
                            for greatgrandchild in grandchild:
                                di[greatgrandchild.tag] = greatgrandchild.text
                                for greatgreatgrandchild in greatgrandchild:
                                    di[greatgreatgrandchild.tag] = greatgreatgrandchild.text
                                    for greatgreatgreatgrandchild in greatgreatgrandchild:
                                        di[greatgreatgreatgrandchild.tag] = greatgreatgreatgrandchild.text
                    di = {k[32:]:v for k,v in di.items()}
                    all_df = pd.concat([all_df, pd.DataFrame([di])], ignore_index=True)
        return all_df

def _get_paper_split(dset_path: str, aid_max: int=1259411, cid_max:int=132472079):
    """
    aid_max = 1259411 # AID 1 to AID 1259411
    cid_max = 132472079 # CID 1 to CID 132472079
    """
    from clamp.dataset import InMemoryClamp
    bb_data = InMemoryClamp(
        root=dset_path,
        assay_mode='',
        compound_mode='smiles',
    )
    comp_df = bb_data.compound_names
    assay_df = bb_data.assay_names
    activity_df = bb_data.activity_df

    aidx_max = len(assay_df[assay_df.AID<=aid_max])
    cidx_max = len(comp_df[comp_df.CID<=cid_max])

    len(assay_df[assay_df.AID<=aid_max])/len(assay_df), len(comp_df[comp_df.CID<=cid_max])/len(comp_df)
    logger.info(f'Reproducing paper-v1.0 split with AID-max: {aidx_max} and CID-max {cidx_max}')
    logger.info(f'Assay split: {len(assay_df[assay_df.AID<=aid_max])/len(assay_df)*100:2.3f}% of assays')
    logger.info(f'Compound split: {len(comp_df[comp_df.CID<=cid_max])/len(comp_df)*100:2.3f}% of compounds')

    paper_set = activity_df.compound_idx<=cidx_max
    paper_set = paper_set & (activity_df.assay_idx<=aidx_max)

    bb_data.num_assays = aidx_max
    bb_data.num_compounds = cidx_max
    bb_data._find_splits() #60-20-20 split
    train_idx = bb_data.subset(
        c_high=bb_data.compound_cut['train'] - 1,
        a_high=bb_data.assay_cut['train'] - 1
    )
    valid_idx = bb_data.subset(
        c_low=bb_data.compound_cut['train'],
        c_high=bb_data.compound_cut['valid'] - 1,
        a_low=bb_data.assay_cut['train'],
        a_high=bb_data.assay_cut['valid'] - 1
    )
    test_idx = bb_data.subset(
        c_low=bb_data.compound_cut['valid'],
        a_low=bb_data.assay_cut['valid'],
    )
    activity_df.loc[train_idx, 'time_a_c_v1'] = 'train'
    activity_df.loc[valid_idx, 'time_a_c_v1'] = 'valid'
    activity_df.loc[test_idx, 'time_a_c_v1'] = 'test'
    logger.info(f'time_a_c_v1 uses {100-activity_df.time_a_c_v1.isna().sum()/len(activity_df)*100:2.3f}% of the data')

    train_idx = bb_data.subset(
        a_high=bb_data.assay_cut['train'] - 1
    )
    valid_idx = bb_data.subset(
        a_low=bb_data.assay_cut['train'],
        a_high=bb_data.assay_cut['valid'] - 1
    )
    test_idx = bb_data.subset(
        a_low=bb_data.assay_cut['valid'],
    )
    activity_df.loc[train_idx, 'time_a_v1'] = 'train'
    activity_df.loc[valid_idx, 'time_a_v1'] = 'valid'
    activity_df.loc[test_idx, 'time_a_v1'] = 'test'

    logger.info(f'time_a_v1 uses {100-activity_df.time_a_c_v1.isna().sum()/len(activity_df)*100:2.3f}% of the data')
    return activity_df

if __name__ == '__main__':
    # parese arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/pubchem23/')
    # cores
    parser.add_argument('--cores', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--skip_download', action='store_true', default=False)
    args = parser.parse_args()

    PUBCHEM_DIR = args.data_dir
    if args.skip_download:
        print('Skip downloading PubChem dataset')
    else:
        download_pubchem(PUBCHEM_DIR)
    
    all_zipfiles = []
    pubchem_assay_dir = os.path.join(PUBCHEM_DIR,'ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/')
    for zipfilenames in os.listdir(pubchem_assay_dir):
        # check if it is a zip file
        if zipfilenames.endswith('.zip'): 
            all_zipfiles.append(os.path.join(pubchem_assay_dir, zipfilenames))

    df = prepro_all_bioassay(all_zipfiles[8], ret_all=False) #quick test
 
    # parallelize the prepro_bioassay function
    num_cores = args.cores
    #all_df = Parallel(n_jobs=num_cores)(delayed(prepro_all_bioassay)(f) for f in all_zipfiles)
    all_df = process_map(prepro_all_bioassay, all_zipfiles, max_workers=num_cores, chunksize=1)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df = all_df.rename(columns={'cid':'CID','aid':'AID'})
    all_df['AID'] = all_df['AID'].astype(int)
    all_df['CID'] = all_df['CID'].astype(int)
    all_df = all_df.sort_values(['AID','CID']) #takes a while ;)
    all_df = all_df.reset_index(drop=True)

    compound_names = all_df['CID'].drop_duplicates().reset_index()[['CID']]
    compound_names.to_parquet(os.path.join(PUBCHEM_DIR, 'compound_names.parquet'))

    # add compound_idx to all_df
    all_df = all_df.join(compound_names.reset_index().set_index('CID').rename(columns={'index':'compound_idx'}), on='CID')

    assay_names = all_df['AID'].drop_duplicates().reset_index()[['AID']]
    assay_names.to_parquet(os.path.join(PUBCHEM_DIR, 'assay_names.parquet'))

    all_df = all_df.join(assay_names.reset_index().set_index('AID').rename(columns={'index':'assay_idx'}), on='AID')

    # save it
    activity_df = all_df[['compound_idx','assay_idx','activity', 'activity_numeric']]
    activity_df.to_parquet(os.path.join(PUBCHEM_DIR, 'activity.parquet'))

    all_zipfiles = []
    pubchem_assay_dir = os.path.join(PUBCHEM_DIR,'ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Description/')
    for zipfilenames in os.listdir(pubchem_assay_dir):
        # check if it is a zip file
        if zipfilenames.endswith('.zip'): 
            all_zipfiles.append(os.path.join(pubchem_assay_dir, zipfilenames))
            
    all_df_description = process_map(prepro_all_bioassay, all_zipfiles, max_workers=50, chunksize=1)
    all_df_description = pd.concat(all_df_description, ignore_index=True)
    all_df_description['ID_id'] = all_df_description.ID_id.astype(int)
    all_df_description = all_df_description.sort_values('ID_id') #takes a while ;)
    all_df_description = all_df_description.reset_index()

    rename_clms = {'ID_id':'AID', 
                'AssayDescription_name':'title', 
                'AssayDescription_description_E':'subtitle',
                'AssayDescription_comment_E': 'comment',
                'AssayTargetInfo_name': 'target_info',
                'AssayTargetInfo_descr': 'target_descr',
                'ConcentrationAttr_concentration': 'concentration',
                'CategorizedComment_comment_E': 'type',
                'AssayDescription_project-category': 'project_category',
                }
    descr_df = all_df_description.rename(columns=rename_clms)[rename_clms.values()]

    assay_names = assay_names.join(descr_df.set_index('AID'), on='AID')
    assay_names.to_parquet(os.path.join(PUBCHEM_DIR, 'assay_names.parquet'))
    #descr_df.to_parquet(os.path.join(PUBCHEM_DIR, 'assay_names.parquet'))

    p = os.path.join(PUBCHEM_DIR, 'ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz')
    df_cid_smi = pd.read_csv(p, compression='gzip', sep='\t', header=None, names=['cid', 'smiles'])

    compound_names = compound_names.join(df_cid_smi.set_index('cid'), on='CID')
    compound_names = compound_names.rename(columns={'smiles':'CanonicalSMILES'})
    compound_names.to_parquet(os.path.join(PUBCHEM_DIR, 'compound_smiles.parquet'))

    # get the paper split
    activity_df = _get_paper_split(PUBCHEM_DIR)
    activity_df.to_parquet(os.path.join(PUBCHEM_DIR, 'activity.parquet'))