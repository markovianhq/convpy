from functools import partial
import numpy as np
from math import isclose
import pandas as pd
from random import random
from scipy.optimize import check_grad
from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import expit as sigmoid
from convpy.functions import (
    hazard, conv_prob, grad_conv_prob, grad_hazard, log_likelikood_conversion, grad_w_c_log_likelikood_conversion,
    grad_w_d_log_likelikood_conversion, grad_w_c_log_likelikood_non_conversion, grad_w_d_log_likelikood_non_conversion,
    objective_function_factory, grad_objective_function_factory
)
from convpy.utils import prepare_input


def test_hazard_and_hidden(simple_feature_matrix):
    X = simple_feature_matrix
    Id = np.identity(3)

    # Exp( first column of X )
    assert np.allclose(hazard(X, Id[0]), np.exp([0, 1, 2, 0]))
    # Exp ( second column of X )
    assert np.allclose(hazard(X, Id[1]), np.exp([1, 0, 5, 2]))
    # Exp ( third column of X )
    assert np.allclose(hazard(X, Id[2]), np.exp([3, 1, -2, -2]))

    # sigmoid ( first column of X )
    assert np.allclose(conv_prob(X, Id[0]), sigmoid(np.asarray([0, 1, 2, 0])))
    # sigmoid ( second column of X )
    assert np.allclose(conv_prob(X, Id[1]), sigmoid(np.asarray([1, 0, 5, 2])))
    # sigmoid ( third column of X )
    assert np.allclose(conv_prob(X, Id[2]), sigmoid(np.asarray([3, 1, -2, -2])))


def test_grad_hazard_and_hidden(simple_feature_matrix):
    X = simple_feature_matrix
    Id = np.identity(3)
    diagonal = lil_matrix((X.shape[0], X.shape[0]))
    diagonal_ = lil_matrix((1, 1))

    assert np.allclose(grad_hazard(X, Id[0], diagonal).toarray(),
                       np.array([grad_hazard(X[i, :], Id[0], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(grad_hazard(X, Id[1], diagonal).toarray(),
                       np.array([grad_hazard(X[i, :], Id[1], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(grad_hazard(X, Id[2], diagonal).toarray(),
                       np.array([grad_hazard(X[i, :], Id[2], diagonal_).toarray()[0] for i in range(4)]))

    assert np.allclose(grad_conv_prob(X, Id[0], diagonal).toarray(),
                       np.array([grad_conv_prob(X[i, :], Id[0], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(grad_conv_prob(X, Id[1], diagonal).toarray(),
                       np.array([grad_conv_prob(X[i, :], Id[1], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(grad_conv_prob(X, Id[2], diagonal).toarray(),
                       np.array([grad_conv_prob(X[i, :], Id[2], diagonal_).toarray()[0] for i in range(4)]))


def test_conversion_part(simple_feature_matrix):
    X = simple_feature_matrix
    Id = np.identity(3)

    a = log_likelikood_conversion(X, Id[0], Id[1], [1, 2, 0, 1])
    b = -np.log(1 + np.exp(-np.asarray([0, 1, 2, 0])))
    b += np.asarray([1, 0, 5, 2]) - np.exp([1, 0, 5, 2]) * [1, 2, 0, 1]

    assert np.allclose(a, b)


def test_grad_log_likelikood_conversion(simple_feature_matrix):
    X = simple_feature_matrix
    Id = np.identity(3)
    time_delta = np.asarray([0.5, 2, 1, 5])
    diagonal = lil_matrix((X.shape[0], X.shape[0]))
    diagonal_ = lil_matrix((1, 1))

    dc_f = grad_w_c_log_likelikood_conversion
    dd_f = grad_w_d_log_likelikood_conversion

    assert np.allclose(dc_f(X, Id[0], diagonal).toarray(),
                       np.array([dc_f(X[i, :], Id[0], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(dc_f(X, Id[1], diagonal).toarray(),
                       np.array([dc_f(X[i, :], Id[1], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(dc_f(X, Id[2], diagonal).toarray(),
                       np.array([dc_f(X[i, :], Id[2], diagonal_).toarray()[0] for i in range(4)]))

    assert np.allclose(dd_f(X, Id[0], time_delta, diagonal).toarray(),
                       np.array([dd_f(X[i, :], Id[0], time_delta[i], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(dd_f(X, Id[1], time_delta, diagonal).toarray(),
                       np.array([dd_f(X[i, :], Id[1], time_delta[i], diagonal_).toarray()[0] for i in range(4)]))
    assert np.allclose(dd_f(X, Id[2], time_delta, diagonal).toarray(),
                       np.array([dd_f(X[i, :], Id[2], time_delta[i], diagonal_).toarray()[0] for i in range(4)]))


def test_grad_log_likelikood_non_conversion(simple_feature_matrix):
    X = simple_feature_matrix
    Id = np.identity(3)
    dt = np.asarray([0.5, 2, 1, 5])
    diagonal = lil_matrix((X.shape[0], X.shape[0]))
    diagonal_ = lil_matrix((1, 1))

    dc_f = grad_w_c_log_likelikood_non_conversion
    dd_f = grad_w_d_log_likelikood_non_conversion

    def to_vector(matrix):
        vector = matrix.toarray()[0]
        return vector

    assert np.allclose(dc_f(X, Id[0], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dc_f(X[i, :], Id[0], Id[0], dt[i], diagonal_)) for i in range(4)]))
    assert np.allclose(dc_f(X, Id[1], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dc_f(X[i, :], Id[1], Id[0], dt[i], diagonal_)) for i in range(4)]))
    assert np.allclose(dc_f(X, Id[2], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dc_f(X[i, :], Id[2], Id[0], dt[i], diagonal_)) for i in range(4)]))

    assert np.allclose(dd_f(X, Id[0], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dd_f(X[i, :], Id[0], Id[0], dt[i], diagonal_)) for i in range(4)]))
    assert np.allclose(dd_f(X, Id[1], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dd_f(X[i, :], Id[1], Id[0], dt[i], diagonal_)) for i in range(4)]))
    assert np.allclose(dd_f(X, Id[2], Id[0], dt, diagonal).toarray(),
                       np.array([to_vector(dd_f(X[i, :], Id[2], Id[0], dt[i], diagonal_)) for i in range(4)]))


def test_objective_function_output_type(test_data_input):
    p = test_data_input['X'].shape[1]

    objective_function = objective_function_factory(fit_intercept=False)

    input_ = prepare_input(**test_data_input)

    w = np.zeros(2 * p)
    input_['Jacobian'] = np.zeros(2 * p)

    assert isinstance(objective_function(w, **input_), float)


def test_objective_function_zero_and_one_weights():
    end_time = 10 * random()
    df = pd.DataFrame([[0, np.nan, 1], [end_time * random(), np.nan, 1],
                       [0, end_time * random(), 1], [0, end_time * random(), 1]],
                      columns=['click_time', 'conv_time', 'intercept'])
    features = ['intercept']
    end_time = 10

    y = df[['click_time', 'conv_time']].values
    X = csr_matrix(df[features].values)

    objective_function = objective_function_factory(fit_intercept=False)

    input_ = prepare_input(y, X, end_time=end_time)
    input_['Jacobian'] = np.zeros(2)

    y0, y1 = input_['y0'], input_['y1']

    w_zero = np.zeros(2)
    w_one = np.ones(2)

    a0_zero = -np.sum(np.log(1 / 2 + 1 / 2 * np.exp(-(end_time - y0[:, 0]))))
    a1_zero = y1.shape[0] * np.log(2) + (y1[:, 1] - y1[:, 0]).sum()
    a0_one = -np.sum(np.log(1 - sigmoid(1) + sigmoid(1) * np.exp(- np.exp(1) * (end_time - y0[:, 0]))))
    a1_one = - y1.shape[0] * (np.log(sigmoid(1)) + 1) + np.exp(1) * (y1[:, 1] - y1[:, 0]).sum()

    assert np.allclose(objective_function(w_zero, **input_), a0_zero + a1_zero)
    assert np.allclose(objective_function(w_one, **input_), a0_one + a1_one + 1.)


def test_grad_objective_function_output_type(test_data_input):
    p = test_data_input['X'].shape[1]

    grad_objective_function = grad_objective_function_factory(fit_intercept=False)

    input_ = prepare_input(**test_data_input)

    w = np.zeros(2 * p)
    input_['Jacobian'] = np.zeros(2 * p)

    assert isinstance(grad_objective_function(w, **input_), np.ndarray)


def test_grad_objective_function_infinite_end_time():
    df = pd.DataFrame([[0, np.nan, 1], [random(), np.nan, 1],
                       [0, random(), 1], [0, random(), 1]],
                      columns=['click_time', 'conv_time', 'intercept'])

    end_time = 1000000

    y = df[['click_time', 'conv_time']].values
    X = csr_matrix(df[['intercept']].values)

    grad_objective_function = grad_objective_function_factory(fit_intercept=False)

    input_ = prepare_input(y, X, end_time=end_time)
    input_['Jacobian'] = np.zeros(2)

    y1 = input_['y1']

    w_zero = np.zeros(2)
    w_one = np.ones(2)

    assert np.allclose(grad_objective_function(w_zero, **input_),
                       np.array([0, (y1[:, 1] - y1[:, 0]).sum() - 2]))
    assert np.allclose(grad_objective_function(w_one, **input_),
                       np.array([df.shape[0] * sigmoid(1) - y1.shape[0],
                                 np.exp(1) * (y1[:, 1] - y1[:, 0]).sum() - 2]) + w_one)


def test_grad_objective_function_infinite_end_time_two_features():
    df = pd.DataFrame([[0, np.nan, 1, random()], [random(), np.nan, 1, random()],
                       [0, random(), 1, random()], [0, random(), 1, random()]],
                      columns=['click_time', 'conv_time', 'intercept', 'random'])
    features = ['intercept', 'random']
    end_time = 1000000

    y = df[['click_time', 'conv_time']].values
    X = csr_matrix(df[features].values)

    grad_objective_function = grad_objective_function_factory(fit_intercept=False)

    input_ = prepare_input(y, X, end_time=end_time)
    input_['Jacobian'] = np.zeros(4)

    y1 = input_['y1']
    x0, x1 = input_['x0'], input_['x1']

    w_zero = np.zeros(4)
    input_['mu'] = 0

    assert np.allclose(grad_objective_function(w_zero, **input_),
                       np.array([0, 1 / 2 * x0[:, 1].sum() - 1 / 2 * x1[:, 1].sum(),
                                 (y1[:, 1] - y1[:, 0] - 1).sum(),
                                 ((y1[:, 1] - y1[:, 0] - 1) * x1[:, 1]).sum()]))


def test_scipy_check_grad(test_dummied_matrix):

    y, X = test_dummied_matrix

    end_time = 1.1 * y[:, 1][~np.isnan(y[:, 1])].max()

    for fit_intercept in [True, False]:
        objective_function = objective_function_factory(fit_intercept=fit_intercept)
        grad_objective_function = grad_objective_function_factory(fit_intercept=fit_intercept)

        input_ = prepare_input(y, X, end_time=end_time)
        p = input_['x0'].shape[1]
        input_['Jacobian'] = np.zeros(2 * p)
        input_['mu'] = 0.
        for i in range(20):
            w0 = (5. - i) * np.ones(2 * p)
            L = partial(objective_function, **input_)
            DL = partial(grad_objective_function, **input_)

        assert isclose(check_grad(L, DL, w0), 0.)
