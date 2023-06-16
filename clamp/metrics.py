
from loguru import logger
from sklearn import metrics
from typing import List, Type

from mlflow.entities import RunInfo
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

def get_sparse_data(m, i):
    return [m.data[index] for index in range(m.indptr[i], m.indptr[i + 1])]

def swipe_threshold_sparse(targets, scores, verbose=True, ret_dict=False):
    """
    Compute ArgMaxJ, AUROC, AVGP and NegAVGP (and more if ret_dict=True) metrics for the true binary values
    `targets` given the predictions `scores`.

    `targets` and `scores` must be :class:`scipy.sparse.csc_matrix` instances
    of the same shape, with rows corresponding to samples and columns to
    variables, so that each metric is computed column-wise.

    Columns being zero everywhere, or having only positive or only negative
    samples need to be skipped, because their classification metrics can not be
    computed.

    In binary classification tasks, given a cut-off value for the predicted
    scores, `Youden's index`_ is defined as

    .. math::
        J = \\text{sensitivity} + \\text{specificity} - 1,

    and ArgMaxJ is the cut-off value which yields highest index. This can be
    interpreted as the cut-off value yielding the "top-left-est" point in the
    ROC curve.

    AUROC_ is the area under the ROC curve. AVGP_ is an approximation of the
    area under the precision-recall curve. The precision-recall curve focuses
    only on the positive class ("precision" is a synonym for "positive
    predictive value", and "recall" is a synonym for "true positive rate").

    NegAVGP (name is made up) is the analogous of the AVGP but for the negative
    class. That is, it is the area under the
    "negative predictive value"-"true negative rate" curve. If the negative
    class is coded by a :math:`0` and the positive class is coded by a
    :math:`1`, then NegAVGP can be computed as the AVGP of the "re-coded"
    labels :math:`y` and predictions :math:`\\hat{y}`:

    .. math::
        \\text{NegAVGP} = \\text{AVGP} (1 - y, 1 - \\hat{y}).

    .. _`Youden's index`: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    .. _AUROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score
    .. _AVGP: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score

    Parameters
    ----------
    targets: :class:`scipy.sparse.csc_matrix`, shape (N, M)
        True target values.
    scores: :class:`scipy.sparse.csc_matrix`, shape (N, M)
        Predicted values.
    verbose: bool
        Be verbose if True.

    Returns
    -------
    tuple of dict
        - ArgMaxJ of each valid column keyed by the column index
        - AUROC of each valid column keyed by the column index
        - AVGP of each valid column keyed by the column index
        - NegAVGP of each valid column keyed by the column index
    """

    assert targets.shape == scores.shape, '"targets" and "scores" must have the same shape.'

    if verbose:
        logger.info('Compute ArgMaxJ, AUROC, AVGP and NegAVGP.')

    # find non-empty columns
    # (https://mike.place/2015/sparse/ for CSR, but works for CSC, too)
    non_empty_idx = np.where(np.diff(targets.indptr) != 0)[0]

    counter_invalid = 0
    argmax_j, auroc, avgp, neg_avgp, bedroc, davgp, dneg_avgp = {}, {}, {}, {}, {}, {}, {}
    bedroc_alpha = 20

    dauprc = {} #check if they are the same as davgp

    for col_idx in non_empty_idx:

        y_true = np.array(list(get_sparse_data(targets, col_idx)))
        if len(pd.unique(y_true)) == 1:  # `pd.unique` is faster than `np.unique` and `set`!
            counter_invalid += 1
            continue
        y_score = np.array(list(get_sparse_data(scores, col_idx)))
        assert len(y_true) == len(y_score)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        assert len(fpr) == len(tpr) == len(thresholds), 'Length mismatch between "fpr", "tpr" and "thresholds".'
        argmax_j[col_idx] = thresholds[np.argmax(tpr - fpr)]

        auroc[col_idx] = metrics.roc_auc_score(y_true, y_score)
        avgp[col_idx] = metrics.average_precision_score(y_true, y_score)
        neg_avgp[col_idx] = metrics.average_precision_score(1 - y_true, 1 - y_score)
        davgp[col_idx] = avgp[col_idx] - y_true.mean()
        dneg_avgp[col_idx] = neg_avgp[col_idx] - (1-y_true.mean())

        bedroc[col_idx] = bedroc_score(y_true, y_score, alpha=bedroc_alpha) #TODO consider also custom alpha

        #precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        #dauprc[col_idx] = metrics.auc(precision, recall)

        #epsilon = (avgp[col_idx]-dauprc[col_idx])
        
    if verbose:
        logger.info(f'Found {len(auroc)} columns with both positive and negative samples.')
        logger.info(f'Found and skipped {counter_invalid} columns with only positive or negative samples.')

    if ret_dict:
        return {'auroc':auroc, 'avgp':avgp, 'neg_avgp':neg_avgp, 
        'argmax_j':argmax_j, 'bedroc':bedroc, 'davgp':davgp, 'dneg_avgp': dneg_avgp}

    return argmax_j, auroc, avgp, neg_avgp, 

def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """BEDROC metric implemented according to Truchon and Bayley.
    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.
    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.
    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.
    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
     """

    assert len(y_true) == len(y_pred), \
        'The number of scores must be equal to the number of labels'

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0] + 1

    s = np.sum(np.exp(-alpha * m_rank / big_n))

    r_a = n / big_n

    rand_sum = r_a * (1 - np.exp(-alpha))/(np.exp(alpha/big_n) - 1)

    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha/2 - alpha * r_a))

    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))

    return s * fac / rand_sum + cte