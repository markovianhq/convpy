from itertools import product
import numpy as np
from numpy.random import exponential
import pandas as pd
from random import random
from scipy.sparse import csr_matrix

from convpy.estimation import ConversionEstimator


def main():
    conv_prob_list = [0.1, 0.2, 0.3, 0.4]
    events = 1000
    lambda_list = [1 / 8, 1 / 4, 1 / 2, 1., 2.]
    end_time = 1.

    for conv_prob_, scale_ in product(conv_prob_list, lambda_list):
        number_of_simulations = 100
        predicted_probability_list_1 = []
        predicted_probability_list_2 = []

        for i in range(number_of_simulations):
            conv_lags_1 = [exponential(scale=scale_) * int(conv_prob_ - random() > 0) for _ in range(events)]
            conv_lags_2 = [exponential(scale=2. * scale_) * int(conv_prob_ / 2. - random() > 0) for _ in range(events)]
            df = pd.DataFrame([[0., t, 1, 0] for t in conv_lags_1] + [[0., t, 0, 1] for t in conv_lags_2],
                              columns=['click_time', 'conv_time', 'f_1', 'f_2'])

            df['conv_time'] = df['conv_time'].map(lambda t: t if t > 0 else np.nan)
            mc_1 = df[(df['f_1'] == 1) & (df['conv_time'].notnull())].shape[0] / df[df['f_1'] == 1].shape[0]
            mc_2 = df[(df['f_2'] == 1) & (df['conv_time'].notnull())].shape[0] / df[df['f_2'] == 1].shape[0]

            # measured data
            df['conv_time'] = df['conv_time'].map(lambda t: t if t <= end_time and not np.isnan(t) else np.nan)

            y = df[['click_time', 'conv_time']].values
            X = csr_matrix(df[['f_1', 'f_2']].values)

            clf = ConversionEstimator(end_time=end_time)
            clf.fit(X, y)

            assert clf.convergence_info['success']
            assert clf.convergence_info['message'] in {b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
                                                       b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'}

            predicted_probability_1 = clf.predict(np.array([[1, 0]]))[0]
            predicted_probability_list_1.append(predicted_probability_1)

            predicted_probability_2 = clf.predict(np.array([[0, 1]]))[0]
            predicted_probability_list_2.append(predicted_probability_2)

        f_1 = pd.DataFrame(predicted_probability_list_1, columns=['predicted conversion probability'])
        f_2 = pd.DataFrame(predicted_probability_list_2, columns=['predicted conversion probability'])

        print('conv_probability: {}, lambda: {}, conversion rate: {}'.format(conv_prob_, 1 / scale_, mc_1))
        print(f_1.describe())
        l = 1  # NOQA
        while abs(f_1.mean()[0] - conv_prob_) >= l * f_1.std()[0]:
            l += 1  # NOQA
        print(" |mean_pred_conv_prob - conv_prob| < {} * std_pred_conv_prob".format(l))

        print('conv_probability: {}, lambda: {}, conversion rate: {}'.format(conv_prob_ / 2, 1 / (3 * scale_), mc_2))
        print(f_2.describe())
        l = 1  # NOQA
        while abs(f_2.mean()[0] - conv_prob_ / 2) >= l * f_2.std()[0]:
            l += 1  # NOQA
        print(" |mean_pred_conv_prob - conv_prob| < {} * std_pred_conv_prob".format(l))

        print('----------')


if __name__ == '__main__':
    main()
