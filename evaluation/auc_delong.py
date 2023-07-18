#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUC statistics using DeLong's test
Fork from:
https://github.com/yandexdataschool/roc_comparison, and
https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals

Also see Stack Overflow Question for further details
https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals

Examples
--------

::

    y_scores = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])

    auc, auc_var, auc_ci = auc_ci_delong(y_true, y_scores, alpha=.95)

    np.sqrt(auc_var) * 2
    max(auc_ci) - min(auc_ci)

    print('AUC: %s' % auc, 'AUC variance: %s' % auc_var)
    print('AUC Conf. Interval: (%s, %s)' % tuple(auc_ci))

    Out:
        AUC: 0.8 AUC variance: 0.028749999999999998
        AUC Conf. Interval: (0.4676719375452081, 1.0)

Reference
----------
Sun and Xu, Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves, IEEE Signal Processing Letters, 21, 1389--1393, 2014

"""
import numpy as np
from scipy import stats


def compute_midrank(x, sample_weight=None):
    """
    Computes midranks with weights.

    Parameters
    ----------
        x : np.array
        sample_weight : np.array
    Returns
    -------
        res2 : np.array, array of midranks
    """
    ind_sort = np.argsort(x)
    x_sort = x[ind_sort]
    len_x = len(x)
    cumulative_weight = np.cumsum(sample_weight[ind_sort]) if sample_weight else np.arange(1, len_x + 1, dtype=float)

    res = np.zeros(len_x, dtype=np.float)
    i = 0
    while i < len_x:
        j = i
        while j < len_x and x_sort[j] == x_sort[i]:
            j += 1
        res[i:j] = cumulative_weight[i:j].mean()
        i = j
    res2 = np.empty(len_x, dtype=np.float)
    res2[ind_sort] = res

    return res2


def fast_delong(predictions_sorted, label_1_count, sample_weight=None):
    """
    The fast version of DeLong's method for computing the covariance of unadjusted AUC.

    Parameters
    ----------
        predictions_sorted : np.array
           a 2D numpy.array[n_classifiers, n_examples] sorted such as the examples with label "1" are first
        label_1_count : number of positive samples
        sample_weight: np.array
    Returns
    -------
        aucs : 1d array, AUC values
        delong_cov : 2d array, DeLong covariance

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted.shape[1] - m
    positive_pred = predictions_sorted[:, :m]
    negative_pred = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    if sample_weight:
        pos_weight, neg_weight = sample_weight[:m], sample_weight[m:]
        total_pos_weight, total_neg_weight = pos_weight.sum(), neg_weight.sum()
        pair_weight = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
        total_pair_weight = pair_weight.sum()
    else:
        pos_weight, neg_weight = None, None
        total_pos_weight, total_neg_weight = m, n
        total_pair_weight = None

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_pred[r, :], pos_weight)
        ty[r, :] = compute_midrank(negative_pred[r, :], neg_weight)
        tz[r, :] = compute_midrank(predictions_sorted[r, :], sample_weight)

    if sample_weight:
        aucs = (pos_weight * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weight
    else:
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / total_neg_weight
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_pos_weight
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def calc_pvalue(aucs, sigma):
    """
    Computes log(10) of p-values.

    Parameters
    ----------
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances

    Returns
    -------
       log10(pvalue) and z value
    """
    z = (aucs[1] - aucs[0]) / (sigma[0, 0] + sigma[1, 1] - 2 * sigma[1, 0])
    return np.log10(2) + stats.norm.logsf(np.abs(z), loc=0, scale=1) / np.log(10), z


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    ordered_sample_weight = None if sample_weight is None else sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Parameters
    ----------
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        sample_weight: np.array, sample weights
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)

    assert len(aucs) == 1, "There is a bug in the code, please forward this to the devs"
    return aucs[0], delong_cov


def delong_roc_test(ground_truth, pred_one, pred_two, sample_weight=None):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different

    Parameters
    ----------
        ground_truth: np.array of 0 and 1
        pred_one: np.array
           predictions of the first model, np.array of floats of the probability of being class 1
        pred_two: np.array
           predictions of the second model, np.array of floats of the probability of being class 1
        sample_weight: np.array
            sample weights
    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth, sample_weight)

    predictions_sorted = np.vstack((pred_one, pred_two))[:, order]
    aucs, delong_cov = fast_delong(predictions_sorted, label_1_count, sample_weight)

    return calc_pvalue(aucs, delong_cov)


def auc_ci_delong(y_true, y_scores, alpha=.95, return_std=True):
    """AUC confidence interval via DeLong.

    Computes de ROC-AUC with its confidence interval via delong_roc_variance

    Parameters
    ----------
    y_true : list
        Ground-truth of the binary labels (allows labels between 0 and 1).
    y_scores : list
        Predicted scores.
    alpha : float
        Default 0.95
    return_std : bool
        Default true, whether to return std or var

    Returns
    -------
        auc : float
            AUC
        auc_var : float
            AUC Variance
        auc_ci : tuple
            AUC Confidence Interval given alpha

    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Get AUC and AUC variance
    auc, auc_var = delong_roc_variance(y_true, y_scores)
    auc_std = np.sqrt(auc_var)

    # Confidence Interval
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    lower_upper_ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    lower_upper_ci[lower_upper_ci > 1] = 1

    if return_std:
        return auc, auc_std, lower_upper_ci
    else:
        return auc, auc_var, lower_upper_ci

# EOF
