#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ref: scipy.stats._resampling

Compute a two-sided bootstrap confidence interval of a statistic according to the following procedure.

1. Resample the data: for each sample in `data` and for each of `n_resamples`,
    take a random sample of the original sample (with replacement) of the same size as the original sample.
2. Compute the bootstrap distribution of the statistic: for each set of resamples, compute the test statistic.
3. Determine the confidence interval: find the interval of the bootstrap distribution that is
   - symmetric about the median and
   - contains `confidence_level` of the resampled statistic values.
"""
from collections import namedtuple
from functools import partial
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats.distributions import norm, chi2, f
from dataclasses import make_dataclass


def _percentile_of_score(a, score, axis):
    """Vectorized, simplified `scipy.stats.percentileofscore`."""
    b = a.shape[axis]
    return np.sum(a < score, axis=axis) / b


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = "The bootstrap distribution is degenerate; the confidence interval is not defined."
            warnings.warn(msg)
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])
# fields = ['confidence_interval', 'standard_error']
BootstrapResult = make_dataclass("BootstrapResult", ['ci', 'se'])


class Bootstrap:
    r"""
    Parameters
    ----------
    data : array-like, sampled from an underlying distribution.
    func : callable
        Statistic for which the confidence interval is to be calculated. `statistic` must be a callable that accepts
        ``len(data)`` samples as separate arguments and returns the resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword argument `axis` and be
        vectorized to compute the statistic along the provided `axis`.
    n_bootstrap : int, default: ``5000``, number of resamples performed to form the bootstrap distribution.
    vectorized : bool, default: ``False``
        If `vectorized` is set ``False``, `statistic` will not be passed keyword argument `axis`,
        and is assumed to calculate the statistic only for 1D samples.
    axis : int, default: ``0``
        The axis of the samples in `data` along which the `statistic` is calculated.
    confidence_level : float, default: ``0.95``
        The confidence level of the confidence interval.
    method : {'percentile', 'basic', 'bca'}, default: ``'BCa'``
        Whether to return the 'percentile' bootstrap confidence interval (``'percentile'``),
        the 'reverse' or the bias-corrected and accelerated bootstrap confidence interval (``'BCa'``).
        Note that only ``'percentile'`` and ``'basic'`` support multi-sample stats at this time.
    random_seed : {None, int}, optional
        Pseudorandom number generator state used to generate resamples.
        If `random_state` is ``None`` (or `np.random`), the `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used, seeded with `random_state`.

    Returns
    -------
    res : BootstrapResult, an object with attributes:

        confidence_interval : ConfidenceInterval
            Bootstrap confidence interval as an instance of `collections.namedtuple` with attributes `low` and `high`.
        standard_error : float or ndarray
            Bootstrap standard error, that is, the sample standard deviation of the bootstrap distribution

    """
    def __init__(
            self, data, func, n_bootstrap=5000, vectorized=False, axis=0, confidence_level=0.95,
            method='BCa', random_seed=0, n_jobs=0
    ):
        # config
        self._n_bootstrap = int(n_bootstrap)
        self._vectorized = bool(vectorized)

        self._alpha = 1 - float(confidence_level)
        self._ci_method = method.lower()
        self._rand_seed = int(random_seed)
        self._n_jobs = int(n_jobs)

        methods = {'percentile', 'basic', 'bca'}
        assert self._ci_method in methods, f"`method` must be in {methods}"

        # states, set sample axis = 0
        assert data.shape[axis] > 1, "`data` must contain two or more observations along `axis`."
        self.data = np.moveaxis(np.asarray(data), axis, 0)  # n_sample * ...
        self.stat_func = partial(func, axis=0) if self._vectorized else func
        self._rng = np.random.RandomState(self._rand_seed)

        # results
        self.theta_ = self.stat_func(self.data)
        self.bootstrap_indices_ = self._bootstrap_resample()
        self.bootstrap_values_ = self.compute_values(self.bootstrap_indices_)
        self.sd_ = np.std(self.bootstrap_values_, ddof=1, axis=0)
        self.ci_ = ConfidenceInterval(*self.compute_ci(self._ci_method))

    @property
    def _n_sample(self):
        return self.data.shape[0]

    @property
    def results(self):
        return BootstrapResult(ci=self.ci_, se=self.sd_)

    def _bootstrap_resample(self):
        return self._rng.randint(
            0, high=self._n_sample, size=(self._n_bootstrap, self._n_sample), dtype=np.uint32
        )

    def compute_values(self, indexes):
        if self._vectorized:
            resamples = self.data[indexes, ...]  # n_boot * n_sample * ...
            theta_boot = np.squeeze(self.stat_func(resamples, axis=1))  # n_boot * n_res
        else:
            if isinstance(self._n_jobs, (int, np.integer)) and self._n_jobs > 1:
                theta_boot_list = Parallel(n_jobs=self._n_jobs)(
                    delayed(self.stat_func)(self.data[index_k, ...]) for index_k in indexes
                )
            else:
                theta_boot_list = [
                    self.stat_func(self.data[index_k, ...]) for index_k in indexes
                ]
            theta_boot = np.asarray(theta_boot_list)  # n_boot * ...
        return theta_boot

    def compute_ci(self, method):
        # confidence intervals
        if method == 'bca':
            interval = self._bca_quantile()
            percentile_fun = _percentile_along_axis
        else:
            interval = self._alpha / 2, 1 - self._alpha / 2
            percentile_fun = partial(np.percentile, axis=0)

        # Calculate confidence interval of statistic
        ci_l = percentile_fun(self.bootstrap_values_, interval[0] * 100)
        ci_u = percentile_fun(self.bootstrap_values_, interval[1] * 100)
        if method == 'basic':  # see [3]
            ci_l, ci_u = 2 * self.theta_ - ci_u, 2 * self.theta_ - ci_l
        return ci_l, ci_u

    def _jackknife_resample(self):
        # jackknife - each row leaves out one observation
        _n = self._n_sample
        jj = np.ones((_n, _n), dtype=bool)
        np.fill_diagonal(jj, False)
        ii = np.arange(_n)
        ii = np.broadcast_to(ii, (_n, _n))
        ii = ii[jj].reshape((_n, _n - 1))
        return ii

    def compute_jackknife_values(self):
        jackknife_indexes = self._jackknife_resample()
        return self.compute_values(jackknife_indexes)

    def _bca_quantile(self):
        """Bias-corrected and accelerated interval."""
        # closely follows [2] "BCa Bootstrap CIs"
        # calculate z0_hat
        percentile = _percentile_of_score(self.bootstrap_values_, self.theta_, axis=0)
        z0_hat = norm.ppf(percentile)

        # calculate a_hat
        theta_hat_i = self.compute_jackknife_values()
        theta_hat_dot = theta_hat_i.mean(axis=0, keepdims=True)
        num = ((theta_hat_dot - theta_hat_i) ** 3).sum(axis=-1)
        den = 6 * ((theta_hat_dot - theta_hat_i) ** 2).sum(axis=0) ** (3 / 2)
        a_hat = num / den

        # calculate alpha_1, alpha_2
        z_alpha = norm.ppf(self._alpha / 2)
        num1, num2 = z0_hat + z_alpha, z0_hat - z_alpha
        alpha_1 = norm.cdf(z0_hat + num1 / (1 - a_hat * num1))
        alpha_2 = norm.cdf(z0_hat + num2 / (1 - a_hat * num2))
        return alpha_1, alpha_2


def compare_pairwise(bootstrap_dict):
    # Pairwise comparison 
    names = list(bootstrap_dict)
    res_comp_df = pd.DataFrame(
        np.nan, columns=names,
        index=pd.MultiIndex.from_product([['p_value', 'z_value'], names])
    )
    for k, name1 in enumerate(names[:-1]):
        for name2 in names[k + 1:]:
            # compare name1, name2
            p_val, z_val = pairwise_test(bootstrap_dict[name1], bootstrap_dict[name2])
            # save
            res_comp_df.loc[('p_value', name1), name2] = p_val
            res_comp_df.loc[('p_value', name2), name1] = p_val
            res_comp_df.loc[('z_value', name1), name2] = z_val
            res_comp_df.loc[('z_value', name2), name1] = -z_val
    for name in names:
        res_comp_df.loc[('p_value', name), name] = 1
        res_comp_df.loc[('z_value', name), name] = 0

    return res_comp_df


def compare_multi_groups(bootstrap_array, bootstrap_axis=0, estimates=None):
    # Multi group comparison, using Hotelling's T-squared statistics
    # H0: x1=x2=...=x_p
    if isinstance(bootstrap_array[0], Bootstrap):
        x_boot_values = np.array([boot.bootstrap_values_ for boot in bootstrap_array])
        estimates = np.array([boot.theta_ for boot in bootstrap_array])
    else:
        assert estimates is not None
        x_boot_values = np.moveaxis(bootstrap_array, bootstrap_axis, -1)

    num_p, num_b = x_boot_values.shape
    dof = num_p - 1
    # let: y1=x2-x1, ..., y[p-1]=x[p]-x[1], then H0: y1=...=y[p-1]=0
    y_boot_values = x_boot_values[1:] - x_boot_values[0]
    y_estimates = np.reshape(estimates[1:] - estimates[0], [-1, 1])
    if np.max(np.abs(y_estimates)) < 1e-8:
        return 1.0, 0.0

    # svd of y-y_mean, y-mean_y = u * s * vh, then Cov(y) = u * s^2 * uT / (n-1)
    mean_y = np.mean(y_boot_values, axis=1, keepdims=True)
    u_y, s_y, _ = np.linalg.svd(y_boot_values - mean_y, full_matrices=False)
    # t2 = y_mean^T*inv(Cov(y))*y_mean = (n-1) * ||(uT*y_mean)/s||^2
    tmp = (u_y.T @ y_estimates).flatten() / s_y
    t2 = np.sum(tmp ** 2) * (num_b - 1)
    # when num_b >> num_p, t2 ~ chi2(num_p-1) under H0
    if num_b > 1000:
        p_val = chi2(df=dof).sf(t2)
    else:
        f_val = t2 * (num_b - dof) / dof / (num_b - 1)
        p_val = f(dfn=dof, dfd=num_b - dof).sf(f_val)
    return p_val, t2


def pairwise_test(boot1: Bootstrap, boot2: Bootstrap, alternative='two_sided'):
    # pairwise comparison between two stats from same samples, using bootstrap z test
    theta_diff = boot1.theta_ - boot2.theta_
    if np.abs(theta_diff) < 1e-8:
        return 1.0, 0.0

    boot_values = boot1.bootstrap_values_ - boot2.bootstrap_values_
    var_boot = np.nanvar(boot_values.flatten(), ddof=1)
    assert np.isfinite(var_boot) and var_boot > 0, f"Invalid bootstrap variance: {var_boot}"
    z_val = theta_diff / np.sqrt(var_boot)
    p_val = p_from_z(z_val, alternative)
    return p_val, z_val


def p_from_z(z, alternative: str = 'two_sided'):
    alternatives = ('less', 'greater', 'two_sided', 'one_sided')
    assert alternative in alternatives, f"alternative must be one of {alternatives}"

    if alternative.lower() == "less":
        p = norm.cdf(z)
    elif alternative.lower() == "greater":
        p = norm.sf(z)
    elif alternative.lower() == "two_sided":
        p = 2 * norm.cdf(-np.abs(z))
    else:
        # one sided
        return p_from_z(z, 'greater') if z > 0 else p_from_z(z, 'less')
    return p

# EOF
