from itertools import chain
import os
from random import random
from uuid import uuid4

import numpy as np
from numpy.random import exponential
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from convpy.functions import conv_prob, hazard
from convpy.preprocessing import OneHotEncoderCOO
from convpy.utils import hash_all
from convpy.conversion_model import ConversionModel

from convpy.estimation import ConversionEstimator
from convpy.utils import prepare_input


@pytest.fixture
def test_data_input():
    path = os.path.dirname(__file__)
    input_ = os.path.join(path, 'data/test_conv_logs.csv')

    df = pd.read_csv(input_, encoding='utf-8')
    features = ['int_feat_1', 'int_feat_2', 'int_feat_3']
    df = df[df[features].notnull().all(axis=1)]
    df = df[(df['conv_time'] - df['click_time'] >= 0) | (df['conv_time'].isnull())]

    y = df[['click_time', 'conv_time']].values
    X = csr_matrix(df[features])

    return {'y': y, 'X': X, 'end_time': 7858471}


@pytest.fixture
def test_df():
    path = os.path.dirname(__file__)
    input_ = os.path.join(path, 'data/test_conv_logs.csv')

    df = pd.read_csv(input_, encoding='utf-8')
    features = ['cat_feat_1', 'cat_feat_2', 'cat_feat_3']

    df = df[df[features].notnull().all(axis=1)]
    df = df[(df['conv_time'] - df['click_time'] >= 0) | (df['conv_time'].isnull())]
    df = df[['click_time', 'conv_time'] + features]

    return df


@pytest.fixture
def test_dummied_matrix(test_df):
    time_data, X = test_df.iloc[:, :2], test_df.iloc[:, 2:]

    X = X.apply(lambda x: hash_all(x, 3), axis=1)
    X = X.astype('str')
    X = pd.get_dummies(X, sparse=True)

    return time_data.values, csr_matrix(X)


@pytest.fixture
def simple_feature_matrix():
    return csr_matrix(np.asarray([[0, 1, 3],
                                  [1, 0, 1],
                                  [2, 5, -2],
                                  [0, 2, -2]]))


@pytest.fixture
def feature_data_factory(tmpdir):

    def factory(features, no_events, modulo=None, default_modulo=100):
        modulo = modulo or {}
        path = tmpdir.join('events_{}.txt'.format(uuid4().hex))

        header = ','.join(features)

        with open(str(path), 'w') as f:
            f.write(header)
            f.write('\n')

            for event_id in range(no_events):
                line = ','.join(chain(
                    (str(hash(uuid4().hex[:6]) % modulo.get(f, default_modulo)) for f in features)
                ))
                f.write(line)
                f.write('\n')

        return str(path)

    return factory


@pytest.fixture
def click_conversion_data_factory():

    def factory(event_rate_weight, conv_prob_weight, feature_value_list, no_events):

        simulator = ConversionModel(feature_value_list, conv_prob_weight, event_rate_weight)
        simulator.sample(samples=no_events * 10 + 1000)

        df = pd.DataFrame(simulator.data.trace())

        conv = simulator.conv.trace()
        lag = simulator.lag.trace()
        lag[conv == 0] = np.nan

        columns = ['click_time', 'conv_time'] + list(df.columns)
        shape_ = lag.shape[0], 1

        df = pd.DataFrame(np.concatenate([np.zeros(shape_), lag.reshape(shape_), df], axis=1), columns=columns)

        return df

    return factory


def generate_random_click_conversion_times(x, end_time, conv_prob_weight_vector, event_rate_weight_vector):
    n, p = x.shape

    click_time = end_time * np. random.rand(n)

    event_rate = hazard(x, event_rate_weight_vector)
    conv_prob_ = conv_prob(x, conv_prob_weight_vector)

    conv_lag = exponential(scale=event_rate) * (conv_prob_ - random() > 0)

    conv_time = click_time + conv_lag
    conv_time[conv_lag <= 0] = np.nan

    return np.array([click_time, conv_time]).T


@pytest.fixture
def estimator_encoder_factory():

    def get_fitted_estimator_and_encoder():
        df = pd.DataFrame([
            [0., np.nan, random()],
            [random(), np.nan, random()],
            [0., random(), random()],
            [0., random(), random()]],
            columns=['click_time', 'conv_time', 'random_feature'])

        features = ['random_feature']
        end_time = 1.
        y, X = df[['click_time', 'conv_time']].values, df[features].values
        X_enc = csr_matrix(X)
        input_ = prepare_input(y, X_enc, end_time=end_time)

        input_['Jacobian'] = np.array([random(), random(), random(), random()])

        estimator = ConversionEstimator(end_time=end_time)
        estimator.fit(X_enc, y)

        encoder = OneHotEncoderCOO(features=['random_feature'])
        encoder.fit(X)

        return estimator, encoder

    return get_fitted_estimator_and_encoder
