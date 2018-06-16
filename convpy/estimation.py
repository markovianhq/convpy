from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # NOQA

from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc

from convpy.functions import conv_prob, hazard, objective_function_factory, grad_objective_function_factory
from convpy.utils import prepare_input


class ConversionEstimator:

    def __init__(self, end_time, fit_intercept=False, **kwargs):
        self.convergence_info = None
        self.coef_ = None
        self.lag_coef_ = None
        self.end_time = end_time
        self.fit_intercept = fit_intercept
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):

        utils_input = (y, X, self.end_time)
        opt_input = prepare_input(*utils_input)

        self.convergence_info = self._get_optimization_result(**opt_input)
        self._get_optimal_params(self.convergence_info)

        return self

    def _get_optimization_result(self, method='L-BFGS-B', **kwargs):
        p = kwargs['x0'].shape[1]
        kwargs['Jacobian'] = np.zeros(2 * p)
        # todo use better initialization

        objective_function = objective_function_factory(self.fit_intercept)
        grad_objective_function = grad_objective_function_factory(self.fit_intercept)

        def L(w):
            return partial(objective_function, **kwargs)(w)

        def DL(w):
            return partial(grad_objective_function, **kwargs)(w)

        w0 = np.zeros(2 * p)

        info = minimize(L, x0=w0, method=method, jac=DL)

        return info

    def _get_optimal_params(self, info):
        weights = info.get('x')
        w_c, w_d = np.split(weights, 2)

        self.coef_ = w_c
        self.lag_coef_ = w_d

    def predict(self, x):
        if len(x.shape) == 1:
            raise TypeError('Input must be (n, p)-dimensional array.'
                            'You passed a p-dimensional array.')

        w_c = self.coef_

        return conv_prob(x, w_c)

    def predict_lag(self, x):
        if len(x.shape) == 1:
            raise TypeError('Input must be (n, p)-dimensional array.'
                            'You passed a p-dimensional array.')

        w_d = self.lag_coef_

        return hazard(x, -w_d)

    @staticmethod
    def log_loss(y_true, y_pred, eps=1e-15):

        y_pred = np.clip(y_pred, eps, 1 - eps)
        loglike = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        return - np.sum(loglike) / len(loglike)

    def plot_coef(self, encoder, save_path=None, **kwargs):
        feature_coef_dict = OrderedDict()
        feature_index = 0

        for i, f in enumerate(encoder.features):
            for v, j in encoder.keymap[i].items():
                feature_coef_dict[(f, v)] = self.coef_[feature_index + j]
            feature_index += len(encoder.keymap[i])

        df = pd.DataFrame(
            [(k, v) for k, v in feature_coef_dict.items()],
            columns=['feature', 'coefficient']
        )
        self._prepare_plot()
        df.plot(x='feature', y='coefficient', kind='barh', **kwargs)
        plt.show()

        if save_path:
            plt.savefig(save_path)

        return df

    def plot_auc(self, X_test, y_test, save_path=None):

        fpr, tpr, thresholds = roc_curve(np.isfinite(y_test[:, 1]), self.predict(X_test))
        roc_auc = auc(fpr, tpr)

        self._prepare_plot()
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_rate_dist(self, X_test, VaR=0.95, save_path=None):
        df = pd.DataFrame(self.predict(X_test), columns=['rates pdf'])
        temp = df[df['rates pdf'] <= df['rates pdf'].quantile(VaR)]

        self._prepare_plot()
        plt.figure()
        temp.plot(color='darkorange', lw=2, label='predicted rates distribution', kind='kde')
        plt.xlim([0., temp['rates pdf'].max() * 1.1])
        plt.xlabel('predicted rates')
        plt.ylabel('density')
        plt.title('predicted rates distribution')
        plt.legend(loc="upper right")
        plt.show()

        if save_path:
            plt.savefig(save_path)

        return df

    @staticmethod
    def _prepare_plot():
        sns.set('talk', 'darkgrid', 'dark', font_scale=1.5,
                rc={"lines.linewidth": 2, 'grid.linestyle': '-'})

    @staticmethod
    def close_plot():
        plt.close()
