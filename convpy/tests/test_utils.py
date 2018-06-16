import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from convpy.utils import prepare_input
from convpy.preprocessing import OneHotEncoderCOO


def test_prepare_input():
    features = ['int_feat_1', 'int_feat_2', 'int_feat_3', 'cat_feat_1', 'cat_feat_2', 'cat_feat_3']

    df = pd.read_csv('convpy/tests/data/test_conv_logs.csv')

    enc = OneHotEncoderCOO()
    X = enc.transform(df[features].values)
    y = df[['click_time', 'conv_time']].values
    end_time = 10.

    input_ = prepare_input(y, X, end_time)

    assert isinstance(input_['y0'], np.ndarray)
    assert isinstance(input_['y1'], np.ndarray)
    assert isinstance(input_['x0'], csr_matrix)
    assert isinstance(input_['x1'], csr_matrix)
    assert isinstance(input_['diagonal0'], coo_matrix)
    assert isinstance(input_['diagonal1'], coo_matrix)
