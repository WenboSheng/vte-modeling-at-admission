#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCA curves
decision curve analysis
see rmda manual: http://mdbrown.github.io/rmda/
rmda codes: https://github.com/mdbrown/rmda
msk tutorial: https://www.mskcc.org/departments/epidemiology-biostatistics/biostatistics/decision-curve-analysis
another python implementation: https://blog.csdn.net/qq_48321729/article/details/123241746
ggDCA: https://mp.weixin.qq.com/s/dcN1BvmuSO7osWFPPq3pYg
"""
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import metrics


def calc_net_benefit(y_label, y_prob, threshold_values=None, with_bootstrap=False, **kwargs):
    """
    Calculate net benefit values, ref: R/subroutines.R in R rmda package
    """
    assert isinstance(y_prob, (pd.Series, np.ndarray)) or threshold_values is not None

    cut_vals = np.sort(np.unique(y_prob)) if threshold_values is None else np.array(threshold_values)
    cut_vals = cut_vals[(cut_vals >= 0) & (cut_vals < 1)]
    n, n_pos, n_neg = len(y_label), np.sum(y_label == 1), np.sum(y_label == 0)
    rho = n_pos / n
    cut_vals_odds = cut_vals / (1 - cut_vals)

    if isinstance(y_prob, str) and y_prob == 'all':
        nb_vals = rho - cut_vals_odds * (1 - rho)
    else:
        nb_vals = np.array(
            [_calc_each_nb(cutoff, y_label, y_prob) for cutoff in cut_vals]
        )

    res_df = pd.DataFrame(
        {'threshold': cut_vals, 'net_benefit': nb_vals, 'sNB': nb_vals / rho}
    )
    if with_bootstrap:
        nb_boot_stat_df, _ = calc_bootstrap_dca(cut_vals, y_label, y_prob, **kwargs)
        snb_boot_stat_df = nb_boot_stat_df / rho
        nb_boot_stat_df.columns = ['nb_std', 'nb_ci_low', 'nb_ci_high']
        snb_boot_stat_df.columns = ['sNB_std', 'sNB_ci_low', 'sNB_ci_high']
        res_df = pd.concat([res_df, nb_boot_stat_df, snb_boot_stat_df], axis=1)

    return res_df


def _calc_each_nb(cutoff, y_label, y_prob):
    n = len(y_label)
    cutoff_odds = cutoff / (1 - cutoff)
    y_pred = y_prob >= cutoff
    _, fp, _, tp = metrics.confusion_matrix(y_label, y_pred).ravel()
    nb = tp / n - fp / n * cutoff_odds
    return nb


def calc_bootstrap_dca(cut_vals, y_label, y_prob, n_boot=5000, ci_alpha=0.05, random_seed=0, n_jobs=0):
    pos_indexes = y_label[y_label == 1].index
    neg_indexes = y_label[y_label != 1].index
    n_pos, n_neg = len(pos_indexes), len(neg_indexes)

    rng = np.random.RandomState(random_seed)
    randint_pos = rng.randint(n_pos, size=(n_boot, n_pos), dtype=np.uint32)
    randint_neg = rng.randint(n_neg, size=(n_boot, n_neg), dtype=np.uint32)

    def calc_bootstrap_metric(rand_pos_index, rand_neg_index):
        rand_indexes = pos_indexes[rand_pos_index].tolist() + neg_indexes[rand_neg_index].tolist()

        y_boot, pred_boot = y_label.loc[rand_indexes], y_prob.loc[rand_indexes]
        nb_vals = np.array(
            [_calc_each_nb(cutoff, y_boot, pred_boot) for cutoff in cut_vals]
        )
        return nb_vals

    nb_boot_list = Parallel(n_jobs=n_jobs)(
        delayed(calc_bootstrap_metric)(randint_pos[k], randint_neg[k]) for k in range(n_boot)
    )

    # bootstrap details: shape: n_boot * n_cutoffs
    nb_boot_arr = np.array(nb_boot_list)
    # std, ci
    nb_boot_stat_df = pd.DataFrame(
        {
            'std': np.std(nb_boot_arr, axis=0),
            'ci_low': np.percentile(nb_boot_arr, ci_alpha * 100.0 / 2, axis=0),
            'ci_high': np.percentile(nb_boot_arr, 100 - ci_alpha * 100.0 / 2, axis=0)
        },
    )

    return nb_boot_stat_df, nb_boot_arr


def plot_dca_curves(
        y_label, y_prob: pd.DataFrame, ax,
        include_all=True, include_none=True, standardize=False,
        **kwargs
):
    # fig, ax = pl.subplots(figsize=(5, 5))
    plot_values_list = []
    nb_col_name = 'sNB' if standardize else 'net_benefit'
    for model, prob_values in y_prob.items():
        dca_values = calc_net_benefit(y_label, prob_values, **kwargs)
        # add model names
        dca_values['model'] = model
        plot_values_list.append(dca_values)
        ax.plot(dca_values['threshold'], dca_values[nb_col_name], label=model)
        ci_names1, ci_names2 = f"{nb_col_name}_ci_low", f"{nb_col_name}_ci_high"
        if ci_names1 in dca_values.columns and ci_names2 in dca_values.columns:
            ax.fill_between(
                dca_values['threshold'], dca_values[ci_names1], dca_values[ci_names2], alpha=0.2
            )

    treat_all_name, treat_none_name = 'Treat all', 'Treat none'
    if include_all:
        max_prob_value = min(np.max(y_prob.values), 1)
        cut_values = np.linspace(0, max_prob_value, 50, endpoint=False)
        dca_values_all = calc_net_benefit(y_label, 'all', cut_values)
        dca_values_all['model'] = treat_all_name
        plot_values_list.append(dca_values_all)
        ax.plot(dca_values_all['threshold'], dca_values_all[nb_col_name], label=treat_all_name)

    if include_none:
        dca_values_none = pd.DataFrame(
            {'threshold': [0, 1], nb_col_name: [0, 0], 'model': [treat_none_name, treat_none_name]}
        )
        plot_values_list.append(dca_values_none)
        ax.plot([0, 1], [0, 0], linestyle=':', label=treat_none_name)

    plot_values = pd.concat(plot_values_list)
    # default xy limit
    ax.set_xlim(0, 1)
    ax.set_ylim(max(plot_values[nb_col_name].min() - 0.15, -0.4), plot_values[nb_col_name].max() + 0.15)
    # default xy label and legend
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Standardized Net Benefit' if standardize else 'Net Benefit')
    ax.legend(loc='lower left')
    return plot_values

# EOF
