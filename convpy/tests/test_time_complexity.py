from itertools import product
import logging
from random import random
from time import time

import numpy as np
import pymc as pm
import pytest
from scipy.sparse import csr_matrix

from convpy.estimation import ConversionEstimator
from convpy.tests.utils import parse_no_events, parse_no_runs


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def time_pipeline(df, end_time):
    time0 = time()

    features = list(df.columns)
    features.pop(features.index('click_time'))
    features.pop(features.index('conv_time'))

    y_obs = df[['click_time', 'conv_time']].copy()
    y_obs['conv_time'] = y_obs['conv_time'].map(lambda x: x if (x <= end_time and not np.isnan(x)) else np.nan)
    y_obs = y_obs.values
    X = csr_matrix(df[features].values)

    estimator = ConversionEstimator(end_time)

    estimator.fit(X, y_obs)

    assert estimator.convergence_info.success

    logger.info('The convergence message is: {}'.format(estimator.convergence_info.message))

    return time() - time0


@pytest.mark.parametrize('end_time,event_rate_weight,conv_prob_weight',
                         list(product([1e+1],
                                      [1e-10],
                                      [1e-10])))
def test_time_complexity(click_conversion_data_factory, end_time, event_rate_weight, conv_prob_weight):

    factory = click_conversion_data_factory

    geo_features = 3
    time_features = 7
    user_features = 5
    publisher_features = 5

    feature_value_list = [geo_features, time_features, user_features, publisher_features]
    n_features = sum(feature_value_list)

    timing_data = []

    for no_events in parse_no_events():
        no_events = int(no_events)

        time_0_factory = time()
        logger.info('Generating random event data ...')
        conv_list = np.array([random() for _ in range(n_features)])
        event_list = np.array([random() for _ in range(n_features)])

        conv_weights = conv_prob_weight / np.sum(conv_list) * conv_list
        event_rate_weights = event_rate_weight / np.sum(event_list) * event_list

        w_c_var = pm.Normal('w_c_var', conv_weights, [10. for _ in range(n_features)], size=n_features)
        w_d_var = pm.Normal('w_d_var', event_rate_weights, [10. for _ in range(n_features)], size=n_features)

        w_c, w_d = w_c_var.random(), w_d_var.random()

        df = factory(w_d, w_c, feature_value_list, no_events)

        logger.info('... done. Generation took {} seconds'.format(time() - time_0_factory))

        timings = [
            time_pipeline(df, end_time)
            for _ in range(parse_no_runs())
        ]

        average_time = sum(timings) / len(timings)

        logger.info('no events {} average time {}'.format(no_events, average_time))

        timing_data.append((no_events, average_time))

    time_factors = []

    for (events_smaller, time_smaller), (events_bigger, time_bigger) in zip(timing_data[:-1], timing_data[1:]):
        events_factor = events_bigger / events_smaller
        time_factor = time_bigger / time_smaller

        time_factors.append(time_factor)

        assert time_factor - events_factor < 6
