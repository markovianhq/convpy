import numpy as np
import pandas as pd
from random import random
from scipy.sparse import csr_matrix

from convpy.estimation import ConversionEstimator
from convpy.utils import prepare_input


def test_minimize():
    df = pd.DataFrame([
        [0., np.nan, random()],
        [random(), np.nan, random()],
        [0., random(), random()],
        [0., random(), random()]],
        columns=['click_time', 'conv_time', 'random_feature'])

    features = ['random_feature']
    end_time = 1.

    y = df[['click_time', 'conv_time']].as_matrix()
    X = csr_matrix(df[features])

    input_ = prepare_input(y, X, end_time=end_time)
    input_['Jacobian'] = np.array([random(), random(), random(), random()])

    clf = ConversionEstimator(end_time=end_time)
    clf.fit(X, y)

    assert isinstance(clf.coef_, np.ndarray) and isinstance(clf.lag_coef_, np.ndarray)
    assert clf.convergence_info['success']
    assert clf.convergence_info['message'] in {b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
                                               b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'}


def test_minimize_large(test_dummied_matrix):

    y, X = test_dummied_matrix

    end_time = 1.1 * y[:, 1][~np.isnan(y[:, 1])].max()

    input_ = prepare_input(y, X, end_time=end_time)
    input_['Jacobian'] = np.array([random() for _ in range(2 * X.shape[1])])

    clf = ConversionEstimator(end_time=end_time)
    clf.fit(X, y)

    assert isinstance(clf.coef_, np.ndarray) and isinstance(clf.lag_coef_, np.ndarray)
    assert clf.convergence_info['success']
    assert clf.convergence_info['message'] in {b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
                                               b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'}
