#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NRI, net reclassification index
"""
import numpy as np
import pandas as pd

from evaluation.metrics import p_from_z


def calc_nri(positive_cross_tab, negative_cross_tab, alternative='two_sided', return_each_class=False):
    """NRI and its statistical inference"""
    # cross tab with row: old model, columns: new model
    # see R `Hmisc::improveProb` in src `R/rcorrp.cens.s`
    tab1 = np.asarray(positive_cross_tab)
    tab0 = np.asarray(negative_cross_tab)

    n_pos, n_neg = tab1.flatten().sum(), tab0.flatten().sum()
    n_level = len(tab1)
    if n_level == 2:
        n_up_pos, n_down_pos = tab1[0, 1], tab1[1, 0]
        n_up_neg, n_down_neg = tab0[0, 1], tab0[1, 0]
    else:
        n_up_pos, n_down_pos = np.triu(tab1, k=1).sum(), np.tril(tab1, k=-1).sum()
        n_up_neg, n_down_neg = np.triu(tab0, k=1).sum(), np.tril(tab0, k=-1).sum()

    p_up_pos, p_down_pos = n_up_pos / n_pos, n_down_pos / n_pos
    p_up_neg, p_down_neg = n_up_neg / n_neg, n_down_neg / n_neg

    nri_pos, nri_neg = p_up_pos - p_down_pos, p_up_neg - p_down_neg
    nri = nri_pos - nri_neg

    var_nri_pos = (n_up_pos + n_down_pos) / n_pos ** 2 - nri_pos ** 2 / n_pos
    var_nri_neg = (n_up_neg + n_down_neg) / n_neg ** 2 - nri_neg ** 2 / n_neg
    se_nri_pos, se_nri_neg = np.sqrt(var_nri_pos), np.sqrt(var_nri_neg)
    se_nri = np.sqrt(var_nri_pos + var_nri_neg)

    z_val = nri / se_nri
    p_val = p_from_z(z_val, alternative)

    z_val_pos, z_val_neg = nri_pos / se_nri_pos, nri_neg / se_nri_neg
    p_val_pos, p_val_neg = p_from_z(z_val_pos, alternative), p_from_z(z_val_neg, alternative)

    res = pd.Series(
        [nri, se_nri, p_val, z_val],
        index=['value', 'se', 'p_value', 'z_value'], name='NRI'
    )
    res_pos = pd.Series(
        [nri_pos, se_nri_pos, p_val_pos, z_val_pos],
        index=['value', 'se', 'p_value', 'z_value'], name='NRI_Positive'
    )
    res_neg = pd.Series(
        [nri_neg, se_nri_neg, p_val_neg, z_val_neg],
        index=['value', 'se', 'p_value', 'z_value'], name='NRI_Negative'
    )
    if return_each_class:
        res = pd.concat([res, res_pos, res_neg], axis=1)
    return res


def compute_idi(y_data, pred1, pred2, alternative='two_sided'):
    """IDI"""
    pred1_pos, pred1_neg = pred1[y_data == 1], pred1[y_data == 0]
    pred2_pos, pred2_neg = pred2[y_data == 1], pred2[y_data == 0]

    diff_pos = pred1_pos - pred2_pos
    diff_neg = pred1_neg - pred2_neg
    idi = np.mean(diff_pos) - np.mean(diff_neg)

    se_idi = np.sqrt(np.var(diff_pos) + np.var(diff_neg))
    z_val = idi / se_idi
    pval = p_from_z(z_val, alternative)

    return idi, se_idi, pval, z_val

# EOF
