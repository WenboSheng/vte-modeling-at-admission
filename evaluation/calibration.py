#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibration test
"""
import numpy as np
from scipy.stats import chi2
import pandas as pd
# from scipy.stats import binom, binomtest, binom_test
# from sklearn.utils import shuffle


def calibration_test(
    y_true, y_prob, *, n_bins=5, strategy="quantile", robust=False, perturb=1e-6
):
    """
    Hosmer-Lemeshow test to judge the goodness of fit for binary data.
    Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    robust : bool, default=False
        Whether to add additional small noise to y_prob, to keep distinct values

    perturb: Float, default 1e-6
        Perturbation noise for prediction

    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).ravel()

    assert len(y_true) == len(y_prob)
    assert y_prob.min() >= 0 and y_prob.max() <= 1, "y_prob has values outside [0, 1]."
    assert strategy in ("quantile", "uniform")
    assert len(np.unique(y_true)) == 2, f"Provided labels {np.unique(y_true)} not binary."

    y_prob_sorted = np.sort(np.unique(y_prob))
    if robust and len(y_prob_sorted) < len(y_prob) and y_prob.min() > 0 and y_prob.max() < 1:
        noise_range = min(
            np.min(np.diff(y_prob_sorted)), y_prob.min(), 1.0 - y_prob.max(), perturb
        ) / 2
        rng = np.random.default_rng(seed=0)
        y_prob += rng.uniform(-noise_range, noise_range, size=len(y_prob))

    if strategy == "quantile":  # Determine bin edges by distribution of data
        bins = np.percentile(y_prob, np.linspace(0, 1, n_bins + 1) * 100)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_ids = np.searchsorted(bins[1:-1], y_prob)
    bin_sums = np.bincount(bin_ids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(bin_ids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(bin_ids, minlength=len(bins))

    nonzero = bin_total != 0
    bin_sums, bin_true, bin_total = bin_sums[nonzero], bin_true[nonzero], bin_total[nonzero]
    prob_true = bin_true / bin_total
    prob_pred = bin_sums / bin_total

    calib_data = pd.DataFrame(
        {
            'prob_observed': prob_true,
            'prob_expected': prob_pred,
            'n_observed': bin_true,
            'n_expected': bin_sums,
            'n_total': bin_total,
            'sd_prob_observed': np.sqrt(prob_true * (1 - prob_true) / bin_total)
        }
    )

    n_levels = len(bin_total)
    hl_stat = np.sum(np.square(bin_true - bin_sums) / (bin_total * prob_pred * (1 - prob_pred)))
    hl_pval = 1 - chi2.cdf(hl_stat, n_levels - 2)

    return hl_pval, hl_stat, n_levels, calib_data


def plot_calibration(ax, calibration_results, xy_max=None, annotate_place=None):
    p_val = calibration_results[0]
    calib_details = calibration_results[-1]
    ax.errorbar(
        x=calib_details['prob_expected'].values,
        y=calib_details['prob_observed'].values,
        yerr=calib_details['sd_prob_observed'].values,
        fmt='o-'
    )
    if not xy_max:
        xy_max = min([1, 1.2 * calib_details['prob_expected'].max()])

    ax.plot([0, xy_max], [0, xy_max], linestyle=':')
    if not annotate_place:
        annotate_place = (0.1, 0.01)
    ax.annotate(f"P={p_val:.3f}", annotate_place, fontstyle='oblique')

# EOF
