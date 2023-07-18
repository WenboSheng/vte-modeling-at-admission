#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model metric collection

Usage:

group_metrics_train = GroupModelMetric(y_train, prediction_train_df)
group_metrics_test = GroupModelMetric(y_test, prediction_test_df, cutoffs=group_metrics_train.cutoff)

group_metrics_test.bootstrap_statistics()

"""
from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy.stats import distributions
# import shap
# import matplotlib.pyplot as plt
# plt.style.use()
from scipy.special import logit, expit
# from seaborn import barplot, pointplot


def _calc_cutoff_metric(cutoff, y_data, y_pred):
    """compute confusion matrix and metrics based on a cutoff value"""
    metric_headings = [
        'cutoff',
        'TN', 'FP', 'FN', 'TP',
        'Sensitivity', 'Specificity', 'Precision', 'NPV',
        'F1', 'F1(neg)', 'MCC', 'Youden', 'Gmean'
    ]

    _y_pred = np.asarray(y_pred >= cutoff, dtype=float)
    _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_data, _y_pred).ravel()
    _p, _r, _f, _s = metrics.precision_recall_fscore_support(y_data, _y_pred, zero_division=0)
    _mcc = metrics.matthews_corrcoef(y_data, _y_pred)

    res_list = [
        cutoff,
        _tn, _fp, _fn, _tp,
        _r[1], _r[0], _p[1], _p[0],
        _f[1], _f[0], _mcc,
        _r[1] + _r[0] - 1,
        np.sqrt(_r[1] * _r[0])
    ]
    return pd.Series(res_list, index=metric_headings)


class ModelMetric:
    def __init__(self, name, cutoff=None):
        self.model_name = name
        self.type_y = 'binary'
        self.metrics_ = dict()
        self.best_cutoff_ = np.nan if cutoff is None else float(cutoff)

    def fit_prediction(self, y_data, y_prob, update_best_cutoff=False):
        # roc details and auroc
        _fpr, _tpr, _cut = metrics.roc_curve(y_data, y_prob)
        _roc_details_df = pd.DataFrame({'fpr': _fpr, 'tpr': _tpr, 'cutoff': _cut})
        _auc = metrics.roc_auc_score(y_data, y_prob)

        # prc details and auprc, 
        # note: len(_cut) = len(unique(y_prob)), len(_p) = len(_r) = len(_cut)+1
        _p, _r, _cut = metrics.precision_recall_curve(y_data, y_prob)
        _prc_details_df = pd.DataFrame({'precision': _p[:-1], 'recall': _r[:-1], 'cutoff': _cut})
        _auprc = metrics.auc(_r, _p)
        _ap = metrics.average_precision_score(y_data, y_prob)

        # classification details: some common metrics w.r.t cutoffs from prc
        _metric_details_df = pd.Series(_cut).apply(
            _calc_cutoff_metric, args=(y_data, y_prob)
        )

        # find best cutoff, default by Youden
        if update_best_cutoff or np.isnan(self.best_cutoff_):
            _best_metric = _metric_details_df.loc[_metric_details_df['Youden'].idxmax()]
            self.best_cutoff_ = _best_metric['cutoff']

        self.metrics_.update(
            {
                'AUROC': _auc,
                'AUPRC': _auprc,
                'AP': _ap,
                'ROC': _roc_details_df,
                'PRC': _prc_details_df,
                'metric': _metric_details_df,
                'TPR_ROC_CI': None,
                'PPV_PRC_CI': None,
            }
        )

        return self

    @property
    def roc_cutoffs(self):
        return self.metrics_['ROC']['cutoff'].values

    @property
    def pr_cutoffs(self):
        return self.metrics_['PRC']['cutoff'].values

    def get_best_cutoff_metric_values(self, metric_names):
        # find actual cutoff and metric values
        metric_detail = self.metrics_['metric']
        df1 = metric_detail.loc[metric_detail['cutoff'] >= self.best_cutoff_]
        best_metric = df1.iloc[0, :] if len(df1) > 0 else metric_detail.iloc[-1, :]

        res = {
            _name: best_metric[_name] if _name in best_metric.index else self.metrics_[_name]
            for _name in metric_names
        }
        return res

    def get_cutoff_metric_values(self, metric_names, cutoff_values) -> pd.DataFrame:
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        metric_detail = self.metrics_['metric']

        res_list = []
        for each_cutoff in cutoff_values:
            df1 = metric_detail.loc[metric_detail['cutoff'] >= each_cutoff]
            cutoff_metrics = df1.iloc[0, :] if len(df1) > 0 else metric_detail.iloc[-1, :]

            each_cutoff_res = {
                _name: cutoff_metrics[_name] if _name in cutoff_metrics.index else self.metrics_[_name]
                for _name in metric_names
            }
            res_list.append(each_cutoff_res)
        res = pd.DataFrame(res_list)
        return res

    def plot_roc(self, ax, label_pos=None, **kwargs):
        """see sklearn.metrics.RocCurveDisplay"""
        auc, roc_details = self.metrics_['AUROC'], self.metrics_['ROC']

        line_kwargs = {"label": f"{self.model_name} ({auc:0.3f})"}
        line_kwargs.update(**kwargs)

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        ax.plot(roc_details.fpr, roc_details.tpr, **line_kwargs)
        ax.set(xlabel="1-Specificity", ylabel="Sensitivity")

        if self.metrics_['TPR_ROC_CI'] is not None:
            tpr_ci_lower = self.metrics_['TPR_ROC_CI'].iloc[:, 0]
            tpr_ci_upper = self.metrics_['TPR_ROC_CI'].iloc[:, 1]
            ax.fill_between(
                roc_details.fpr, tpr_ci_lower, tpr_ci_upper, alpha=0.2,
            )

        if "label" in line_kwargs and label_pos:
            # "lower right"
            ax.legend(loc=label_pos)

        return ax

    def plot_prc(self, ax, label_pos=None, use_ap=False, **kwargs):
        """see sklearn.metrics.PrecisionRecallDisplay"""
        ap, prc_details = self.metrics_['AP'] if use_ap else self.metrics_['AUPRC'], self.metrics_['PRC']

        line_kwargs = {"drawstyle": "steps-post", "label": f"{self.model_name} ({ap:0.3f})"}
        line_kwargs.update(**kwargs)

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

        recalls, precisions = prc_details['recall'].values[::-1], prc_details['precision'].values[::-1]
        if not (recalls[0] == 0 and precisions[0] == 1):
            recalls = np.hstack((0, recalls))
            precisions = np.hstack((1, precisions))

        ax.plot(recalls, precisions, **line_kwargs)
        ax.set(xlabel="Recall", ylabel="Precision")

        if self.metrics_['PPV_PRC_CI'] is not None:
            ppv_prc_ci = self.metrics_['PPV_PRC_CI']
            prec_ci_lower = ppv_prc_ci.iloc[::-1, 0].values
            prec_ci_upper = ppv_prc_ci.iloc[::-1, 1].values
            if len(ppv_prc_ci) < len(recalls):
                prec_ci_lower = np.hstack((1, prec_ci_lower))
                prec_ci_upper = np.hstack((1, prec_ci_upper))
            ax.fill_between(
                recalls, prec_ci_lower, prec_ci_upper, alpha=0.2,
            )

        if "label" in line_kwargs and label_pos:
            # "lower left"
            ax.legend(loc=label_pos)

        return ax


class GroupModelMetric:
    """A group of model evaluation given a dataset"""
    def __init__(self, y_data, predictions, model_name_map=None, best_cutoffs=None, auto_set_cutoff=False):
        self.y_data = pd.Series(y_data)
        if model_name_map is None:
            self.prediction_df = predictions
        else:
            self.prediction_df = pd.DataFrame(
                {model_name_map[_model]: predictions[_model].values for _model in predictions.columns},
                index=predictions.index
            )
        self.best_cutoffs = defaultdict(float) if best_cutoffs is None else best_cutoffs.copy()

        # compute model metrics
        self.model_metrics_ = {
            _name: ModelMetric(_name).fit_prediction(self.y_data, self.prediction_df[_name])
            for _name in self.prediction_df.columns
        }

        # set cutoff by current dataset
        if auto_set_cutoff:
            self.best_cutoffs = {
                _name: each_metric.best_cutoff_ for _name, each_metric in self.model_metrics_.items()
            }
        self.roc_cutoffs = {
            _name: each_metric.roc_cutoffs for _name, each_metric in self.model_metrics_.items()
        }
        self.pr_cutoffs = {
            _name: each_metric.pr_cutoffs for _name, each_metric in self.model_metrics_.items()
        }

    def get_metric_summary_df(self, metric_names=('AUROC', 'AUPRC', 'Sensitivity', 'Specificity')):
        return pd.DataFrame(
            {
                _model: each_metric.get_best_cutoff_metric_values(metric_names)
                for _model, each_metric in self.model_metrics_.items()
            }
        )

    def bootstrap_statistics(
            self, metric_names=('AUROC', 'AUPRC', 'Sensitivity', 'Specificity'),
            n_boot=5000, ci_alpha=0.05, random_seed=0, n_jobs=1
    ):

        pos_indexes = self.y_data[self.y_data == 1].index
        neg_indexes = self.y_data[self.y_data != 1].index
        n_pos, n_neg = len(pos_indexes), len(neg_indexes)

        rng = np.random.RandomState(random_seed)
        randint_pos = rng.randint(n_pos, size=(n_boot, n_pos), dtype=np.uint32)
        randint_neg = rng.randint(n_neg, size=(n_boot, n_neg), dtype=np.uint32)

        def calc_bootstrap_metric(rand_pos_index, rand_neg_index):
            rand_indexes = pos_indexes[rand_pos_index].tolist() + neg_indexes[rand_neg_index].tolist()

            y_boot, pred_boot = self.y_data.loc[rand_indexes], self.prediction_df.loc[rand_indexes]

            model_metrics_boot = {
                model: ModelMetric(name=model, cutoff=self.best_cutoffs[model])
                for model in self.model_metrics_
            }
            sens_roc_dict = dict()
            prec_pr_dict = dict()
            for model, model_metric in model_metrics_boot.items():
                model_metric.fit_prediction(y_boot, pred_boot[model], update_best_cutoff=False)
                _roc_cutoffs = self.roc_cutoffs[model]
                _sens_boot = model_metric.get_cutoff_metric_values('Sensitivity', _roc_cutoffs).values.flatten()
                _prc_cutoffs = self.pr_cutoffs[model]
                _prec_boot = model_metric.get_cutoff_metric_values('Precision', _prc_cutoffs).values.flatten()
                sens_roc_dict[model] = _sens_boot
                prec_pr_dict[model] = _prec_boot

            metric_values_boot_df = pd.DataFrame(
                {
                    _model: each_metric.get_best_cutoff_metric_values(metric_names)
                    for _model, each_metric in model_metrics_boot.items()
                }
            )
            return metric_values_boot_df.values, sens_roc_dict, prec_pr_dict

        res_boot_list = Parallel(n_jobs=n_jobs)(
            delayed(calc_bootstrap_metric)(randint_pos[k], randint_neg[k]) for k in range(n_boot)
        )

        # bootstrap details: shape: n_boot * n_model * n_metric
        res_boot_arr = np.array([x[0] for x in res_boot_list])
        # std, ci
        std_boot = pd.DataFrame(
            np.std(res_boot_arr, axis=0),
            columns=list(self.model_metrics_), index=list(metric_names)
        )
        ci_lower_boot = pd.DataFrame(
            np.percentile(res_boot_arr, ci_alpha * 100.0 / 2, axis=0),
            columns=list(self.model_metrics_), index=list(metric_names)
        )
        ci_upper_boot = pd.DataFrame(
            np.percentile(res_boot_arr, 100 - ci_alpha * 100.0 / 2, axis=0),
            columns=list(self.model_metrics_), index=list(metric_names)
        )

        # sens of roc curves
        sens_roc_dict_list = [x[1] for x in res_boot_list]
        prec_prc_dict_list = [x[2] for x in res_boot_list]

        for model in self.model_metrics_.keys():
            sens_roc_model = np.array([x[model] for x in sens_roc_dict_list])  # n_boot * x
            sens_roc_ci_lower_boot = np.percentile(
                sens_roc_model, ci_alpha * 100.0 / 2, axis=0
            )
            sens_roc_ci_upper_boot = np.percentile(
                sens_roc_model, 100 - ci_alpha * 100.0 / 2, axis=0
            )
            sens_roc_ci = pd.DataFrame(
                {
                    'Sensitivity_ci_lower': sens_roc_ci_lower_boot,
                    'Sensitivity_ci_upper': sens_roc_ci_upper_boot,
                }
            )

            prec_pr_model = np.array([x[model] for x in prec_prc_dict_list])
            prec_pr_ci_lower_boot = np.percentile(
                prec_pr_model, ci_alpha * 100.0 / 2, axis=0
            )
            prec_pr_ci_upper_boot = np.percentile(
                prec_pr_model, 100 - ci_alpha * 100.0 / 2, axis=0
            )
            prec_pr_ci = pd.DataFrame(
                {
                    'Precision_ci_lower': prec_pr_ci_lower_boot,
                    'Precision_ci_upper': prec_pr_ci_upper_boot,
                }
            )
            self.model_metrics_[model].metrics_.update(
                {
                    'TPR_ROC_CI': sens_roc_ci,
                    'PPV_PRC_CI': prec_pr_ci,
                }
            )

        return std_boot, ci_lower_boot, ci_upper_boot, res_boot_arr

    def plot_roc_curves(self, ax):
        for _model, _metric in self.model_metrics_.items():
            _metric.plot_roc(ax=ax)
        ax.plot([0, 1], [0, 1], '--')


def search_optimal_threshold(classif_metric_df, metric_name, metric_value, tol=1e-2, metric_backup='Youden'):
    df1 = classif_metric_df.loc[classif_metric_df[metric_name] >= metric_value - tol]
    df2 = df1.loc[df1[metric_name] == np.min(df1[metric_name])]
    if len(df2) > 1:
        idx_optimal = df2[metric_backup].idxmax()
    else:
        idx_optimal = df2.index[0]

    best_metric = classif_metric_df.loc[idx_optimal]
    best_cut = best_metric['cutoff']
    best_values = best_metric.iloc[1:]
    return best_cut, best_values


def bootstrap_z_test(estimate, boot_values, alternative='two_sided'):
    """Bootstrap z test for H0: estimate=0"""

    if np.abs(estimate) < 1e-8:
        return 1.0, 0.0
    var_boot = np.nanvar(boot_values.flatten(), ddof=1)
    assert np.isfinite(var_boot) and var_boot > 0, f"Invalid bootstrap variance: {var_boot}"
    z_val = estimate / np.sqrt(var_boot)
    p_val = p_from_z(z_val, alternative)
    return p_val, z_val


def p_from_z(z, alternative: str = 'two_sided'):
    alternatives = ('less', 'greater', 'two_sided', 'one_sided')
    assert alternative in alternatives, f"alternative must be one of {alternatives}"

    if alternative.lower() == "less":
        p = distributions.norm.cdf(z)
    elif alternative == "greater":
        p = distributions.norm.sf(z)
    elif alternative == "two_sided":
        p = 2 * distributions.norm.sf(np.abs(z))
    else:
        return p_from_z(z, 'greater') if z > 0 else p_from_z(z, 'less')
    return p


def compute_binomial_ci(theta, n, alpha=0.05, method='binomial'):
    phi_ci = distributions.norm.ppf(1.0 - alpha / 2.0)
    if method == 'binomial':
        delta = phi_ci * np.sqrt(theta * (1 - theta) / n)
        ci = np.array([theta - delta, theta + delta])
    else:
        eta = logit(theta)
        tau = 1 / np.sqrt(n * theta * (1 - theta))
        eta_ci = [eta - phi_ci * tau, eta + phi_ci * tau]
        ci = expit(eta_ci)
    return ci

# EOF
