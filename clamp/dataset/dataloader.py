from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger
from scipy import sparse
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd

try:
    # only if Graph-Model is used
    import dgl
except: pass

def get_sparse_indices_and_data(m, i):
    return m.indices[m.indptr[i]:m.indptr[i + 1]], m.data[m.indptr[i]:m.indptr[i + 1]]

class InMemoryClamp(Dataset):
    """
    Subclass of :class:`torch.utils.data.Dataset` holding BioBert activity
    data, that is, activity triplets, and compound and assay feature vectors.

    :class:`InMemoryClamp` supports two different indexing (and iteration)
    styles. The default style is to iterate over `(compound, assay, activity)`
    COO triplets, however they are sorted. The "meta-assays" style consists in
    iterating over unique compounds using a CSR sparse structure, and averaging
    the feature vectors of the positive and negative assays of each compound.
    """

    def __init__(
            self,
            root: Union[str, Path],
            assay_mode: str,
            compound_mode: str = None,
            train_size: float = 0.6,
            aid_max: int = None,
            cid_max: int = None,
            verbose: bool = True
    ) -> None:
        """
        Instantiate the dataset class.

        - The data is loaded in memory with the :meth:`_load_dataset` method.
        - Splits are created separately along compounds and along assays with
          the :meth:`_find_splits` method. Compound and assay splits can be
          interwoven with the :meth:`subset` method.

        Parameters
        ----------
        root: str or :class:`pathlib.Path`
            Path to a directory of ready BioBert files.
        assay_mode: str
            Type of assay features ("biobert-last", "biobert-two-last", or "lsa").
        train_size: float (between 0 and 1)
            Fraction of compounds and assays assigned to training data.
        verbose: bool
            Be verbose if True.
        """
        self.root = Path(root)
        self.assay_mode = assay_mode
        self.compound_mode = compound_mode
        self.train_size = train_size
        self.verbose = verbose

        self._load_dataset()
        self._find_splits()

        self.meta_assays = False
        self.assay_onehot = None

    def _load_dataset(self) -> None:
        """
        Load prepared dataset from the `root` directory:

        - `activity`: Parquet file containing `(compound, assay, activity)`
          triplets. Compounds and assays are represented by indices, and thus
          the file is directly loaded into a :class:`scipy.sparse.coo_matrix`
          with rows corresponding to compounds and columns corresponding to
          assays.

        - `compound_names`: Parquet file containing the mapping between the
          compound index used in `activity` and the corresponding compound name.
          It is loaded into a :class:`pandas.DataFrame`.

        - `assay_names`: Parquet file containing the mapping between the assay
          index used in `activity` and the corresponding assay name. It is
          loaded into a :class:`pandas.DataFrame`.

        - `compound_features`: npz file containing the compound features array,
          where the feature vector for the compound indexed by `idx` is stored
          in the `idx`-th row. It is loaded into a
          :class:`scipy.sparse.csr_matrix`.

        - `assay_features`: npy file containing the assay features array, where
          the feature vector for the assay indexed by `idx` is stored in the
          `idx`-th row. It is loaded into a :class:`numpy.ndarray`.

        Compute the additional basic dataset attributes `num_compounds`,
        `num_assays`, `compound_features_size`, `assay_feature_size`.
        """

        if self.verbose:
            logger.info(f'Load dataset from "{self.root}" with "{self.assay_mode}" assay features.')

        with open(self.root / 'compound_names.parquet', 'rb') as f:
            self.compound_names = pd.read_parquet(f)

        compound_modes = self.compound_mode.split('||') if self.compound_mode is not None else 1
        if len(compound_modes)>1:
            logger.info('Multiple compound modes are concatenated')
            self.compound_features = np.concatenate([self._load_compound(cm) for cm in compound_modes], axis=1)
        else:
            self.compound_features = self._load_compound(self.compound_mode)

        self.num_compounds = len(self.compound_names)
        if 'graph' in self.compound_mode and (not 'graphormer' in self.compound_mode):
            self.compound_features_size = self.compound_features[0].ndata['h'].shape[1] #in_edge_feats
        elif isinstance(self.compound_features, pd.DataFrame):
            self.compound_features_size = 40000
        else:
            if len(self.compound_features.shape)>1:
                self.compound_features_size = self.compound_features.shape[1]
            else:
                self.compound_features_size = 1

        with open(self.root / 'assay_names.parquet', 'rb') as f:
            self.assay_names = pd.read_parquet(f)

        self.num_assays = len(self.assay_names)

        assay_modes = self.assay_mode.split('||')
        if len(assay_modes)>1:
            logger.info(f'Multiple assay modes are concatenated')
            self.assay_features = np.concatenate([self._load_assay(am) for am in assay_modes], axis=1)
        else:
            self.assay_features = self._load_assay(self.assay_mode)

        if (self.assay_features is None):
            self.assay_features_size = 512 #wild guess also 512
        elif len(self.assay_features.shape)==1:
            # its only a list, so probably text
            self.assay_features_size = 768
        else:
            self.assay_features_size = self.assay_features.shape[1]

        with open(self.root / ('activity.parquet'), 'rb') as f:
            activity_df = pd.read_parquet(f)
            self.activity_df = activity_df

        self.activity = sparse.coo_matrix(
            (
                activity_df['activity'],
                (activity_df['compound_idx'], activity_df['assay_idx'])
            ),
            shape=(self.num_compounds, self.num_assays)
        )

    def _load_compound(self, compound_mode=None) :
        cmpfn = f'compound_features{"_"+compound_mode if compound_mode else ""}'
        if 'graph' in compound_mode and (not 'graphormer' in compound_mode):
            logger.info(f'graph in compound_mode: loading '+cmpfn)
            import dgl
            from dgl.data.utils import load_graphs
            compound_features = load_graphs(str(self.root/(cmpfn+".bin")))[0]
            compound_features = np.array(compound_features)
        elif compound_mode =='smiles':
            compound_features = pd.read_parquet(self.root/('compound_smiles.parquet'))['CanonicalSMILES'].values
        else:
            try: #tries to open npz file else npy
                with open(self.root / (cmpfn+'.npz'), 'rb') as f:
                    compound_features = sparse.load_npz(f)
            except:
                logger.info(f'loading '+cmpfn+'.npz failed, using .npy instead')
                try:
                    compound_features = np.load(self.root / (cmpfn+'.npy'))
                except:
                    logger.info(f'loading '+cmpfn+'.npy failed, trying to compute it on the fly')
                    compound_features = pd.read_parquet(self.root / ('compound_smiles.parquet'))
        return compound_features

    def _load_assay(self, assay_mode='lsa') -> None:
        """ loads assay """
       # assert assay_mode in [
       #     'biobert-last', 'biobert-two-last', 'lsa', ''
       # ], f'Assay features "{assay_mode}" are not known.'
        if assay_mode=='':
            print('no assay features')
            return None

        if assay_mode == 'biobert-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_last_layer.npy'), 'rb') as f:
                return np.load(f, allow_pickle=True)
        elif assay_mode == 'biobert-two-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_penultimate_and_last_layer.npy'), 'rb') as f:
                return  np.load(f, allow_pickle=True)
        #elif assay_mode == 'lsa':
        #    with open(self.root/('assay_features_lsa.npy'), 'rb') as f:
        #        return np.load(f)
        try: #tries to open npz file else npy
            with open(self.root/(f'assay_features_{assay_mode}.npz'), 'rb') as f:
                return sparse.load_npz(f)
        except:
            with open(self.root/(f'assay_features_{assay_mode}.npy'), 'rb') as f:
                return np.load(f, allow_pickle=True)
        #except FileNotFoundError:
        #    logger.info(f'assay_features_{assay_mode}.npy not found')
        #    return np.nan
        return None

    def _make_angles_dataset(
            self,
            num_compounds: int = 1000,
            num_assays: int = 500,
            random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        """
        Make a synthetic dataset, dubbed "Angles", for debugging.
        Compounds and assays are both represented by 2D features. The points
        lie in a circle of radius 1. The activity data is defined as follows:
        if the angle between a compound feature vector and an assay feature
        vector is smaller than 45 degree, then this compound and this assay are
        positively related; otherwise they are negatively related.

        Parameters
        ----------
        num_compounds: int
            Number of unique compounds.
        num_assays: int
            Number of unique assays.
        random_state: None, int or :class:`numpy.random.RandomState`
            Sets the pseudo-random behaviour.
        """

        from sklearn.utils import check_random_state
        import itertools as it

        rng = check_random_state(random_state)

        if self.verbose:
            logger.info(f'Make "Angles" dataset with {num_compounds} compounds and {num_assays} assays.')

        self.num_compounds = num_compounds
        self.compound_names = list(range(self.num_compounds))

        self.num_assays = num_assays
        self.assay_names = list(range(self.num_assays))

        theta = rng.uniform(low=0, high=2 * np.pi, size=(self.num_compounds + self.num_assays))
        X = np.vstack([[np.cos(t), np.sin(t)] for t in theta])

        compound_features = X[:self.num_compounds]
        self.compound_features = sparse.csr_matrix(compound_features.astype(np.float32))
        self.compound_features_size = self.compound_features.shape[1]

        assay_features = X[self.num_compounds:]
        self.assay_features = assay_features.astype(np.float32)
        self.assay_features_size = self.assay_features.shape[1]

        activity = []
        for c_idx, a_idx in it.product(range(self.num_compounds), range(self.num_assays)):
            cosine = self.compound_features[c_idx].dot(self.assay_features[a_idx])
            if cosine >= np.sqrt(2) / 2:  # angle <= 45 degrees
                activity.append([c_idx, a_idx, 1])
            else:
                activity.append([c_idx, a_idx, 0])
        activity = np.vstack(activity)

        self.activity = sparse.coo_matrix(
            (activity[:, 2], (activity[:, 0], activity[:, 1])),
            shape=(self.num_compounds, self.num_assays)
        )

    @staticmethod
    def _chunk(n: int, first_cut_ratio: float) -> Tuple[int, int]:
        """
        Find the two cut points required to chunk a sequence of `n` items into
        three parts, the first having `first_cut_ratio` of the items, the second
        and the third having approximately the half of the remaining items.

        Parameters
        ----------
        n: int
            Length of the sequence to chunk.
        first_cut_ratio: float
            Portion of items in the first chunk.

        Returns
        -------
        int, int
            Position where the first and second cut must occur.
        """

        first_cut = int(round(first_cut_ratio * n))
        second_cut = first_cut + int(round((n - first_cut) / 2))

        return first_cut, second_cut

    def _find_splits(self) -> None:
        """
        We assume that during the preparation of the PubChem data, compounds
        (assays) have been indexed so that a larger compound (assay) index
        corresponds to a compound (assay) incorporated to PubChem later in time.
        This function finds the compound (assay) index cut-points to create
        three chronologically disjoint splits.

        The oldest `train_size` fraction of compounds (assays) are assigned to
        training. From the remaining compounds (assays), the oldest half are
        assigned to validation, and the newest half are assigned to test.
        Only the index cut points are stored.
        """
        if self.verbose:
            logger.info(f'Find split cut-points for compound and assay indices (train_size = {self.train_size}).')

        first_cut, second_cut = self._chunk(self.num_compounds, self.train_size)
        self.compound_cut = {'train': first_cut, 'valid': second_cut}

        first_cut, second_cut = self._chunk(self.num_assays, self.train_size)
        self.assay_cut = {'train': first_cut, 'valid': second_cut}


    def subset(
            self,
            c_low: Optional[int] = None,
            c_high: Optional[int] = None,
            a_low: Optional[int] = None,
            a_high: Optional[int] = None
    ) -> np.ndarray:
        """
        Find the indices of the `activity` triplets in default, COO style, where
        compounds and assays have index within the given ranges (included),
        that is

        - `c_low <= compound_idx <= c_high` and
        - `a_low <= assay_idx <= a_high`.

        If `*_low` is `None`, set it to :math:`0` (smallest possible index).
        If `*_high` is `None`, set it to `num_compounds` or `num_assays`
        (highest possible index).

        Parameters
        ----------
        c_low: int or None
            Lowest compound index included.
        c_high: int or None
            Highest compound index included.
        a_low: int or None
            Lowest assay index included.
        a_high: int or None
            Highest assay index included.

        Returns
        -------
        :class:`numpy.ndarray` of int
            Subset of indices of the `activity` triplets.

        """
        if c_low is None:
            c_low = 0
        if a_low is None:
            a_low = 0
        if c_high is None:
            c_high = self.num_compounds
        if a_high is None:
            a_high = self.num_assays

        if self.verbose:
            logger.info(f'Find activity triplets where {c_low} <= compound_idx <= {c_high} and {a_low} <= assay_idx <= {a_high}.')

        activity_bool = np.logical_and.reduce(
            (
                self.activity.row >= c_low,
                self.activity.row <= c_high,
                self.activity.col >= a_low,
                self.activity.col <= a_high
            )
        )

        return np.flatnonzero(activity_bool)

    def get_unique_names(
            self,
            activity_idx: Union[int, Iterable[int], slice]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the unique compound and assay names within the `activity` triplets
        indexed by `activity_idx` in default, COO style. The names are returned
        in alphabetical order, not in the order in which they occur in
        `activity`. This is useful to list the unique compounds and assays
        within a dataset split.

        Parameters
        ----------
        activity_idx: int, iterable of int, or slice
            Index to one or multiple `activity` triplets.

        Returns
        -------
        tuple of :class:`pandas.DataFrame`
            - `compound_names`
            - `assay_names`
        """

        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]

        if isinstance(compound_idx, np.ndarray) and isinstance(assay_idx, np.ndarray):
            compound_idx = pd.unique(compound_idx)
            assay_idx = pd.unique(assay_idx)

        elif isinstance(compound_idx, (int, np.integer)) and isinstance(assay_idx, (int, np.integer)):
            pass  # a single integer is already a unique index

        else:
            raise TypeError

        compound_names = self.compound_names.iloc[compound_idx]
        assay_names = self.assay_names.iloc[assay_idx]

        return compound_names.sort_index(), assay_names.sort_index()

    def setup_assay_onehot(self, size: int) -> None:
        """
        Use this function to initialize `assay_onehot` once `size` is known.
        This is a helper function for multitask models.

        Parameters
        ----------
        size: int
            Number of assays that should be considered.
        """
        self.assay_onehot = OneHot(size=size)

    def getitem(
            self,
            activity_idx: Union[int, Iterable[int], slice],
            ret_np=False,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the `activity` triplets indexed by `activity_idx` in default, COO
        style. That is, `len(activity_idx)` triplets like
        `(compound_idx, assay_idx, activity)`.

        Obtain the corresponding feature vectors by querying the
        `compound_features` and `assay_features` arrays at the rows indicated
        by `compound_idx` and `assay_idx`, respectively.

        The `activity_idx` is returned too, to be able to reconstruct the order
        in which the dataset has been visited.

        The `assay_idx` is returned too, mounted as one-hot vectors. This is
        useful for multitask models.

        Parameters
        ----------
        activity_idx: int, iterable of int, or slice
            Index to one or multiple `activity` triplets.

        Returns
        -------
        tuple of :class:`torch.Tensor`, prepended by `activity_idx` in whichever format it was provided.

            - `activity_idx`
            - `compound_features`: shape (len(activity_idx), compound_feat_size),
            - `assay_features`: shape (len(activity_idx), assay_feat_size)
            - `assay_onehot`: shape (len(activity_idx), assay_feat_size)
            - `activity`: shape (len(activity_idx), ).
        """

        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]
        activity = self.activity.data[activity_idx]

        if isinstance(self.compound_features, pd.DataFrame):
            compound_smiles = self.compound_features.iloc[compound_idx]['CanonicalSMILES'].values
            from mhnreact.molutils import convert_smiles_to_fp
            if self.compound_mode == 'MxFP':
                fptype = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
            else:
                fptype = self.compound_mode
            # TODO fp_size as input parameter
            fp_size = 40000
            compound_features = convert_smiles_to_fp(compound_smiles, fp_size=fp_size, which=fptype, radius=2, njobs=1).astype(np.float32)
        else:
            compound_features = self.compound_features[compound_idx]
            if isinstance(compound_features, sparse.csr_matrix):
                compound_features = compound_features.toarray()
        

        # compound_features = compound_features.astype(np.float32)

        assay_features = self.assay_features[assay_idx]
        if isinstance(assay_features, sparse.csr_matrix):
            assay_features = assay_features.toarray()

        try:
            assay_onehot = self.assay_onehot[assay_idx].toarray()
        except (TypeError, ValueError):
            assay_onehot = np.zeros_like(assay_features)

        if isinstance(activity_idx, (int, np.integer)):
            compound_features = compound_features.reshape(-1)
            assay_features = assay_features.reshape(-1)
            assay_onehot = assay_onehot.reshape(-1)
            activity = [activity]
        elif isinstance(activity_idx, list):
            if len(activity_idx) == 1:
                compound_features = compound_features.reshape(-1)
                assay_features = assay_features.reshape(-1)
                assay_onehot = assay_onehot.reshape(-1)
        activity = np.array(activity)


        if ret_np:
            return (
            activity_idx,                         # as is
            compound_features,  # already float32
            assay_features if not isinstance(assay_features[0], str) else assay_features,     # already float32
            assay_onehot if not isinstance(assay_onehot[0], str) else assay_onehot,       # already float32
            (float(activity))    # torch.nn.BCEWithLogitsLoss needs this to be float too...
            )


        if self.compound_mode =='smiles':
            comp_feat = compound_features
        elif isinstance(compound_features, np.ndarray):
            comp_feat = torch.from_numpy(compound_features)
        elif not isinstance(compound_features[0], dgl.DGLGraph):
            comp_feat = dgl.batch(compound_features)
        else:
            comp_feat = compound_features

        return (
            activity_idx,                         # as is
            comp_feat,  # already float32
            torch.from_numpy(assay_features) if not isinstance(assay_features[0], str) else assay_features,     # already float32
            torch.from_numpy(assay_onehot.astype(int)) if not isinstance(assay_onehot[0], str) else assay_onehot,       # already float32
            torch.from_numpy(activity).float()    # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    def getitem_meta_assays(
            self,
            compound_idx: Union[int, List[int], slice],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the `activity` triplets indexed by `compound_idx` in "meta-assays",
        CSR style. It may be a different number of triplets for each row.

        Obtain the corresponding feature vectors by querying the
        `compound_features` and `assay_features` arrays at the rows indicated
        by `compound_idx` and the related `assay_idx`, respectively.

        Average the assay feature vectors related to one compound into
        positives and negatives, so that at most the following two triplets
        result for the compound:

        - :math:`\\left( \\mathbf{x}_c, \\text{avg} \\{\\mathbf{a}_c^+\\}, 1 \\right)`
        - :math:`\\left( \\mathbf{x}_c, \\text{avg} \\{\\mathbf{a}_c^-\\}, 0 \\right)`,

        where :math:`\\mathbf{x}_c` is the compound feature vector,
        :math:`\\{\\mathbf{a}_c^+\\}` are the feature vectors of the related
        positive assays, and :math:`\\{\\mathbf{a}_c^-\\}` are the feature
        vectors of the related negative assays.

        Parameters
        ----------
        compound_idx: int, iterable of int, or slice
            Index to one or multiple compounds.

        Returns
        -------
        tuple of :class:`torch.Tensor`

            - `compound_features`: shape (N, compound_feat_size),
            - `assay_features`: shape (N, assay_feat_size)
            - `activity`: shape (N, ).
        """

        activity_slice = self.activity.tocsr()[compound_idx]

        # find non-empty rows (https://mike.place/2015/sparse/
        non_empty_row_idx = np.where(np.diff(activity_slice.indptr) != 0)[0]

        compound_features_l = []
        assay_positive_features_l, assay_negative_features_l = [], []
        activity_l = []

        for row_idx in non_empty_row_idx:

            positive_l, negative_l = [], []

            for col_idx, activity in get_sparse_indices_and_data(activity_slice, row_idx):
                if activity == 0:
                    negative_l.append(self.assay_features[col_idx])
                else:
                    positive_l.append(self.assay_features[col_idx])

            if len(negative_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                negative = np.vstack(negative_l).mean(axis=0)
                assay_negative_features_l.append(negative)
                activity_l.append(0)

            if len(positive_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                positive = np.vstack(positive_l).mean(axis=0)
                assay_positive_features_l.append(positive)
                activity_l.append(1)

        compound_features = sparse.vstack(compound_features_l).toarray()
        assay_features_l = np.vstack(
            assay_negative_features_l + assay_positive_features_l  # "+" is list concatenation!
        )
        activity = np.vstack(activity_l)

        return (
            torch.from_numpy(compound_features),  # already float32
            torch.from_numpy(assay_features_l),   # already float32
            torch.from_numpy(activity).float()    # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    @staticmethod
    def collate(
            batch_as_list: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Necessary for :meth:`getitem_meta_assays` if using a
        :class:`torch.utils.data.DataLoader`. Not necessary if using
        :class:`torch.utils.data.BatchSampler`, as I typically do.

        Parameters
        ----------
        batch_as_list: list
            Result of :meth:`getitem_meta_assays` for a mini-batch.

        Returns
        -------
        tuple of :class:`torch.Tensor`
            Data for a mini-batch.
        """
        compound_features_t, assay_features_t, activity_t = zip(*batch_as_list)
        return (
            torch.cat(compound_features_t, dim=0),
            torch.cat(assay_features_t, dim=0),
            torch.cat(activity_t, dim=0),
        )

    def __getitem__(
            self,
            idx: Union[int, Iterable[int], slice]
    ) -> Tuple:
        """
        Index or slice `activity` by `idx`. The indexing mode depends on the
        value of `meta_assays`. If False (default), the indexing is over COO
        triplets. If True, the indexing is over unique compounds.
        """

        if self.meta_assays:
            return self.getitem_meta_assays(idx)
        else:
            return self.getitem(idx)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        - If `meta_assays` is False (default), length is defined as the number
          of `(compound, assay, activity)` COO triplets.
        - If `meta_assays` is True, length is defined as the number of unique
          compounds.
        """

        if self.meta_assays:
            return self.num_compounds
        else:
            return self.activity.nnz

    def __repr__(self):
        return f'InMemoryClamp\n' \
               f'\troot="{self.root}"\n' \
               f'\tassay_mode="{self.assay_mode}"\n' \
               f'\ttrain_size={self.train_size}\n' \
               f'\tactivity.shape = {self.activity.shape}\n' \
               f'\tactivity.nnz = {self.activity.nnz}\n' \
               f'\tmeta_assays={self.meta_assays}'


class OneHot:
    """
    Class to create sparse 1-hot vectors at given indices as if a
    :class:`scipy.sparse.csr_matrix` were indexed or sliced. This is a helper
    class for multitask models.
    """
    def __init__(self, size: int) -> None:
        """
        Initialize class.

        Parameters
        ----------
        size: int
            Size of the one-hot vectors.
        """
        self.size = size

    def __getitem__(
            self,
            idx: Union[int, Iterable[int], slice]
    ) -> sparse.csr_matrix:
        """
        Return a sparse array with one-hot vectors as rows.

        Examples
        --------
        >>> oh = OneHot(3)
        >>> oh[[0, 0, 1]].toarray()
        array([[1, 0, 0],
               [1, 0, 0],
               [0, 1, 0]])
        >>> oh[:].toarray()
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

        Parameters
        ----------
        idx: int, Itreable[int], slice
            One or multiple indices.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`, shape (len(idx), size)
            Sparse array with one-hot vectors as rows.
        """
        assert isinstance(idx, (int, np.integer, Iterable, slice)), 'Unknown index type.'

        if isinstance(idx, (int, np.integer)):
            onehot = sparse.csr_matrix(
                (
                    [1],
                    ([0], [idx])
                ),
                shape=(1, self.size),
                dtype=int
            )
        elif isinstance(idx, Iterable):
            onehot = sparse.csr_matrix(
                (
                    np.ones(len(idx)),
                    (np.arange(len(idx)), idx)
                ),
                shape=(len(idx), self.size),
                dtype=int
            )
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.size
            step = idx.step if idx.step is not None else 1

            idx_list = [i for i in range(start, stop, step)]

            onehot = sparse.csr_matrix(
                (
                    np.ones(len(idx_list)),
                    (np.arange(len(idx_list)), idx_list)
                ),
                shape=(len(idx_list), self.size),
                dtype=int
            )
        else:
            onehot = None

        return onehot


if __name__ == '__main__':

    im_biobert = InMemoryClamp('../data/pubchem_tiny', 'lsa')

    dataloader_coo = DataLoader(
        im_biobert,
        batch_size=len(im_biobert),
        shuffle=True,
        num_workers=0
    )
    for k, batch in enumerate(dataloader_coo):
        print(f'batch {k} in COO style')
        for v in batch[1:]:
            print(f'\t{v.shape}')

    im_biobert.meta_assays = True
    dataloader_csr = DataLoader(
        im_biobert,
        batch_size=len(im_biobert),
        shuffle=True,
        num_workers=0,
        collate_fn=im_biobert.collate
    )
    for k, batch in enumerate(dataloader_csr):
        print(f'batch {k} in CSR style')
        for v in batch:
            print(f'\t{v.shape}')
