from clamp.dataset import InMemoryCLAMP, OneHot
import numpy as np


im_clampds = InMemoryCLAMP(
    '../data/pubchem_tiny',
    assay_mode='clip',
    verbose=False
)


def test_chunk():

    first_cut, second_cut = im_clampds._chunk(n=10, first_cut_ratio=0.4)
    assert first_cut == 4
    assert second_cut == 4 + ((10 - 4) / 2)


def test_unique_names():
    # unique names for one row in `activities` must be one compound and one assay
    compounds, assays = im_clampds.get_unique_names(im_clampds.subset(22, 22, 0, 0))
    assert len(compounds) == len(assays) == 1

    # unique names for the whole dataset must agree with the class attributes
    compounds, assays = im_clampds.get_unique_names(im_clampds.subset())
    assert compounds.equals(im_clampds.compound_names)
    assert assays.equals(im_clampds.assay_names)


def test_meta_assays_indexing():
    """
    In `pubchem_tiny`, `activity`, seen as a matrix, has compound 3117 in
    the 0-th row, for which assays 493165 and 493166 are measured.
    >>> im_clampds.get_unique_names(im_clampds.subset(c_low=0, c_high=0))
    >>> (    CID
         0  3117,
               AID
         2  493165
         3  493166)

    This corresponds to indices 20 and 30 in default, COO style.
    >>> im_clampds.subset(c_low=0, c_high=0)
    >>> array([20, 30])

    Both assays are positive.
    >>> im_clampds[[20, 30]][4]
    >>> tensor([1., 1.])
    """

    _, compound_coo, assay_coo, _, activity_coo = im_clampds[[20, 30]]
    assert compound_coo[0].equal(compound_coo[1])

    im_clampds.meta_assays = True
    compound_csr, assay_csr, activity_csr = im_clampds[0]
    assert compound_coo[0].equal(compound_csr.flatten())
    assert assay_coo.mean(dim=0).equal(assay_csr.flatten())


def test_onehot():

    oh = OneHot(3)

    assert np.array_equal(
        oh[[0, 0, 1]].toarray(),
        np.array([[1, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0]])
    )

    assert np.array_equal(
        oh[:].toarray(),
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    )
