from clamp import metrics
from scipy import sparse
import numpy as np


def test_metrics_dense():
    # columns 0 and 2 are valid, columns 1 and 3 are invalid
    targets = np.array(
        [[1, 1, 0, 0],
         [0, 1, 1, 0]]
    )

    # column 0 is a wrong pred, column 2 is a correct pred, columns 1 and 3 don't matter
    scores_or_preds = np.array(
        [[0, 0, 0, 0],
         [1, 0, 1, 0]]
    )

    mask = np.array(
        [[1, 0, 1, 0],
         [1, 0, 1, 0]]
    )

    _, auroc, _, _ = metrics.swipe_threshold_dense(targets, scores_or_preds, verbose=False)  # check auroc only
    assert auroc[0] == 0.
    assert auroc[2] == 1.

    _, auroc, _, _ = metrics.swipe_threshold_dense(targets, scores_or_preds, mask, verbose=False)  # check auroc only
    assert auroc[0] == 0.
    assert auroc[2] == 1.

    accuracy = metrics.fix_threshold_dense(targets, scores_or_preds, verbose=False)
    assert accuracy[0] == 0.
    assert accuracy[2] == 1.

    accuracy = metrics.fix_threshold_dense(targets, scores_or_preds, mask, verbose=False)
    assert accuracy[0] == 0.
    assert accuracy[2] == 1.


def test_metrics_sparse():
    # remember: in my implementation I have explicit 0s, which are kept,
    # but this means that here I need dummy data with explicit 0s as well

    # targets
    row = [0, 0, 0, 0, 1, 1, 1, 1]
    col = [0, 1, 2, 3, 0, 1, 2, 3]
    targets = [1, 1, 0, 0, 0, 1, 1, 0]
    # this yields ...
    # [[1, 1, 0, 0],
    #  [0, 1, 1, 0]],
    targets_csc = sparse.csc_matrix((targets, (row, col)), shape=(2, 4))

    # scores
    # row = [0, 0, 0, 0, 1, 1, 1, 1]
    # col = [0, 1, 2, 3, 0, 1, 2, 3]
    scores_or_preds = [0, 0, 0, 0, 1, 0, 1, 0]
    # this yields ...
    # [[0., 0., 0., 0.],
    #  [1., 0., 1., 0.]],
    # where assay 0 wrong, assay 2 well, assays 1 and 3 don't matter
    scores_or_preds_csc = sparse.csc_matrix((scores_or_preds, (row, col)), shape=(2, 4))

    _, auroc, _, _ = metrics.swipe_threshold_sparse(targets_csc, scores_or_preds_csc, verbose=False)  # check auroc only
    assert auroc[0] == 0.
    assert auroc[2] == 1.

    accuracy = metrics.fix_threshold_sparse(targets_csc, scores_or_preds_csc, verbose=False)
    assert accuracy[0] == 0.
    assert accuracy[2] == 1.
