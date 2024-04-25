
from pathlib import Path
from scipy import sparse
import sys
import numpy as np
import pandas as pd
import argparse
import os
from loguru import logger

"""
Extract compound encoding from a parquet files containing IDs and SMILES strings.
If no compound IDs are provided, a 1:1 mapping is assumed, and only a coumpound2smiles parquet file is required.
The column name for the compound SMILES is assumed to be 'CanonicalSMILES'.
"""

""" example call:
python clamp/dataset/encode_compound.py \
--compounds=./data/moleculenet/tox21/compound_names.parquet \
--compound2smiles=./data/moleculenet/tox21/compound_smiles.parquet \
--fp_type=morganc+rdkc --fp_size=8192


or 
python clamp/dataset/encode_compound.py --compound2smiles=./data/pubchem23/compound_smiles.parquet --compounds=./data/pubchem23/compound_names.parquet --fp_type=morganc+rdkc --fp_size=8192
"""

class SparseMorganEncoder:
    def __init__(self, radius=2, fp_size=1024, njobs=1):
        self.radius = radius
        self.fp_size = fp_size
        self.njobs = njobs
        if fp_size>65535:
            raise ValueError('fp_size must be <= 65535 (uint16) for sparse matrix representation.')

    def encode(self, smiles_list):
        fps = Parallel(njobs=self.njobs)(
            delayed(self._get_morgan_fingerprint)(smiles) for smiles in tqdm(smiles_list)
        )
        return self._sparse_matrix_from_fps(fps)

    def _get_morgan_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.fp_size)
            # if less than 64 bits are on; this is more efficient; otherwise storing the 1024 bits is more efficient
            return np.array((fp.GetOnBits()), dtype=np.uint16) # only valid for fp_size 65535
        else:
            return None

    def _sparse_matrix_from_fps(self, on_bits):
        n_samples = len(on_bits)
        sparse_matrix = sparse.lil_matrix((n_samples, self.fp_size), dtype=bool)
        # lil -- efficient for constructing
        for i, fp in enumerate(on_bits):
            if fp is not None:
                sparse_matrix[np.array([i]*len(fp)), fp] = True
        print('converting to csr for saving')
        return sparse_matrix.tocsr() # fro saving to csr

class CdddEncoder:
    def __init__(self, njobs=32):
        self.njobs = njobs
        self.cddd_dir = "/system/user/seidl/seidl/projects/biobert/model_zoo/cddd/"
        self._default_model_dir = os.path.join(DEFAULT_DATA_DIR, 'default_model')
        self.infer_model = InferenceModel(self._default_model_dir, use_gpu=False, cpu_threads=self.njobs)
        
        # Add CDDD directory for import
        sys.path.append(self.cddd_dir)
        from cddd.inference import InferenceModel
        from cddd.hyperparameters import DEFAULT_DATA_DIR
        
    def encode(self, list_of_smiles):
        X_fp = []
        bs = 2048 #batch_size
        for bi in range(0, len(list_of_smiles), bs):
            X_fp.append(self.infer_model.seq_to_emb(list_of_smiles[bi:min(bi+bs, len(list_of_smiles))]))
        X_fp = np.concatenate(X_fp)
        return X_fp

class ClampEncoder:
    def __init__(self, device='cuda:0'):
        from clamp.models.pretrained import PretrainedCLAMP
        self.model = PretrainedCLAMP()
        self.model.to(device)
        self.model.eval()
        self.device = device

    def encode(self, list_of_smiles):
        X_fp = []
        bs = 2048
        for bi in range(0, len(list_of_smiles), bs):
            batch_smi = list_of_smiles[bi:min(bi+bs, len(list_of_smiles))]
            embedding = self.model.encode_smiles(batch_smi).detach().cpu().numpy()
            X_fp.append(embedding)
        X_fp = np.vstack(X_fp)
        return X_fp

class MLRUNEncoder(ClampEncoder):
    def __init__(self, run_dir, device='cuda:0', compound_features_size=8192, assay_features_size=512):
        from clamp import utils
        run_dir = Path(run_dir)
        model, hparams= utils.load_model(run_dir, 
                                compound_features_size=compound_features_size, 
                                assay_features_size=assay_features_size, 
                                device=device, ret_hparams=True)
        self.model = model
        self.hparams = hparams
        self.model.eval().to(device)


class FpEncoder:
    def __init__(self, fp_size=8192, fp_type='morganc+rdkc', radius=2, njobs=32, disable_logging=True):
        self.fp_size = fp_size
        self.fp_type = fp_type
        self.radius = radius
        self.njobs = njobs
        self.disable_logging = disable_logging

        from mhnreact.molutils import convert_smiles_to_fp
        self.convert_smiles_to_fp = convert_smiles_to_fp
        
        if self.disable_logging:
            from mhnreact.molutils import disable_rdkit_logging
            disable_rdkit_logging()
        
        if self.fp_type == 'MxFP':
            self.fp_type = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
    
    def encode(self, list_of_smiles):
        return self.convert_smiles_to_fp(list_of_smiles, fp_size=self.fp_size, is_smarts=False, which=self.fp_type, radius=self.radius, njobs=self.njobs, verbose=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute RDKit sparse features for a collection of PubChem compounds.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compound2smiles', help='Path to a Parquet file mapping CIDs to SMILES strings.')
    parser.add_argument('--compounds',default=None, help='Path to a Parquet file with compound ids (CIDs) for which to extract features. If none is Provided a 1:1 mapping is assumed.')
    parser.add_argument('--fp_type', help='fp_types, e.g. sprsFP, morganc+rdkc, MxFP, cddd', default='morganc+rdkc', type=str)
    parser.add_argument('--fp_size', help='fp_size', default=8096, type=int)
    parser.add_argument('--njobs', help='njobs', default=50, type=int)
    parser.add_argument('--smiles_column', help='smiles_column', default='CanonicalSMILES', type=str)
    parser.add_argument('--standardize', help='standardize molecules', default=False, type=bool)

    args = parser.parse_args()
    compound2smiles_df = pd.read_parquet(args.compound2smiles)
    # if CID is not in the columns
    if 'CID' not in compound2smiles_df.columns:
        # warn the user
        print('WARNING: CID not in compound2smiles_df.columns, using the index as CID.')
        compound2smiles_df['CID'] = compound2smiles_df.index
    compound2smiles = compound2smiles_df.set_index('CID')[args.smiles_column].to_dict()

    if args.standardize:
        from ..mol_utils import standardize
        import tqdm
        # consider parallelizing this
        compound2smiles = {k:standardize(v) for k,v in tqdm.tqdm(compound2smiles.items(), total=len(compound2smiles), desc='standardizing smiles')}

    if args.compounds is None:
        compounds = list(range(len(compound2smiles)))
        # warning with continue
        #raise Warning('No compound CID to idx provided, assuming 1:1 mapping.')
        print('No compound CID to idx provided, assuming 1:1 mapping.')
    else:
        compounds_df = pd.read_parquet(args.compounds)
        compounds = compounds_df['CID'].squeeze().tolist() #added CID for FSMOL , should still work ;)
 
    logger.info(f'converting {len(compounds)} smiles to features')

    list_of_smiles = [compound2smiles[c] for c in compounds]

    if 'mlruns' in args.fp_type:
        encoder = MLRUNEncoder(args.fp_type) #todo set fp_size and assay_features_size
        args.fp_type = args.fp_type.split('/')[-1]

    if args.fp_type=='sprsFP':
        encoder = SparseMorganEncoder(radius=2, fp_size=args.fp_size, njobs=args.njobs)
    elif args.fp_type=='cddd':
        enocder = CdddEncoder(njobs=args.njobs)
    elif args.fp_type=='clamp':
        logger.info('Using CLAMP encoder')
        enocder = ClampEncoder() #todo path to model
    else:
        enocder = FpEncoder(fp_size=args.fp_size, fp_type=args.fp_type, njobs=args.njobs)
    
    x = enocder.encode(list_of_smiles)
    
    p = Path(args.compound2smiles).with_name(f'compound_features_{args.fp_type}.npy')
    logger.info(f'Save compound features with shape {x.shape} to {p}')
    np.save(p, x) if args.fp_type!='sprsFP' else sparse.save_npz(p, x)


