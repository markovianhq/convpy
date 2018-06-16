from functools import partial
import numpy as np
from scipy.special import expit as sigmoid


def d_sigmoid(t):
    return sigmoid(t) * (1. - sigmoid(t))


def hazard(x, w_d):
    """
    :param x: scipy.sparse.csr.csr_matrix of dimension (n, p)
    :param w_d: np.array of dimension p
    :return: np.array of dimension n
    """
    return np.exp(x.dot(w_d))


def grad_hazard(x, w_d, diagonal):
    """
    Gradient of the hazard function with respect to the weight vector.

    :param x: scipy.sparse.csr.csr_matrix of dimension (n, p)
    :param w_d: np.array of dimension p
    :param diagonal: scipy.sparse.lil_matrix of dimension (n, n)
    :return: scipy.sparse.csr.csr_matrix of dimension (n, p)
    """
    diagonal.setdiag(hazard(x, w_d))

    return diagonal * x


def conv_prob(x, w_c):
    """
    :param x: scipy.sparse.csr.csr_matrix of dimension (n, p)
    :param w_c: np.array of dimension p
    :return: np.array of dimension n
    """
    return sigmoid(x.dot(w_c))


def grad_conv_prob(x, w_c, diagonal):
    """
    Gradient of the conv_prob function with respect to the weight vector.

    :param x: scipy.sparse.csr.csr_matrix of dimension (n, p)
    :param w_c: np.array of dimension p
    :param diagonal: scipy.sparse.coomatrix of dimension (n, n)
    :return: scipy.sparse.csr.csr_matrix of dimension (n, p)
    """
    diagonal.setdiag(d_sigmoid(x.dot(w_c)))

    return diagonal * x


def log_likelikood_conversion(x, w_c, w_d, time_delta):
    return np.log(conv_prob(x, w_c)) + x.dot(w_d) - time_delta * np.exp(x.dot(w_d))


def grad_w_c_log_likelikood_conversion(x, w_c, diagonal):
    diagonal.setdiag((1. - conv_prob(x, w_c)))

    return diagonal * x


def grad_w_d_log_likelikood_conversion(x, w_d, time_delta, diagonal):
    diagonal.setdiag(1. - time_delta * np.exp(x.dot(w_d)))

    return diagonal * x


def log_likelikood_non_conversion(x, w_c, w_d, elapsed_time):
    return np.log(1. - conv_prob(x, w_c) + conv_prob(x, w_c) * np.exp(- elapsed_time * np.exp(x.dot(w_d))))


def grad_w_c_log_likelikood_non_conversion(x, w_c, w_d, elapsed_time, diagonal):
    temp = np.exp(- elapsed_time * np.exp(x.dot(w_d)))
    norm = 1. - conv_prob(x, w_c) + conv_prob(x, w_c) * temp

    diagonal.setdiag(conv_prob(x, w_c) * (1. - conv_prob(x, w_c)) * (temp - 1.) / norm)

    res = diagonal * x

    # norm == 0 implies conv_prob(x, w_c) == 1 and temp != np.inf
    res[norm == 0.] = 0.
    # norm == np.inf implies conv_prob(x, w_c) != 0 and temp == np.inf
    diagonal.setdiag(1. - conv_prob(x, w_c))

    try:
        res[np.isposinf(norm)] = (diagonal * x)[np.isposinf(norm)]
    except ValueError:
        pass

    return res


def grad_w_d_log_likelikood_non_conversion(x, w_c, w_d, elapsed_time, diagonal):
    lambda_elapsed = elapsed_time * np.exp(x.dot(w_d))
    temp = conv_prob(x, w_c) * np.exp(-lambda_elapsed)

    diagonal.setdiag((elapsed_time * temp * np.exp(x.dot(w_d))) / (1. - conv_prob(x, w_c) + temp))

    res = - diagonal * x

    res[(1. - conv_prob(x, w_c) + temp == 0.) & np.isfinite(lambda_elapsed)] = 0.

    return res


def objective_function(w, Jacobian, y0, y1, x0, x1, end_time, diagonal0, diagonal1, mu, penalty):  # NOQA
    # neg_log_likelihood with regularization term
    w_c, w_d = np.split(w, 2)

    a0 = log_likelikood_non_conversion(x0, w_c, w_d, end_time - y0[:, 0])
    T = y1[:, 1] - y1[:, 0]
    a1 = log_likelikood_conversion(x1, w_c, w_d, T)

    return - np.sum(a0) - np.sum(a1) + penalty(w_c, mu) + penalty(w_d, mu)


def grad_objective_function(w, Jacobian, y0, y1, x0, x1, end_time, diagonal0, diagonal1, mu, grad_penalty):
    w_c, w_d = np.split(w, 2)

    p = x0.shape[1]
    n0 = y0.shape[0]
    n1 = y1.shape[0]

    g0c = grad_w_c_log_likelikood_non_conversion(x0, w_c, w_d, end_time - y0[:, 0], diagonal0)
    g1c = grad_w_c_log_likelikood_conversion(x1, w_c, diagonal1)

    g0d = grad_w_d_log_likelikood_non_conversion(x0, w_c, w_d, end_time - y0[:, 0], diagonal0)
    g1d = grad_w_d_log_likelikood_conversion(x1, w_d, y1[:, 1] - y1[:, 0], diagonal1)

    Jacobian[:p] = - np.ones(n0) * g0c - np.ones(n1) * g1c + grad_penalty(w_c, mu)
    Jacobian[p:] = - np.ones(n0) * g0d - np.ones(n1) * g1d + grad_penalty(w_d, mu)

    return Jacobian


def objective_function_factory(fit_intercept):

    if fit_intercept:
        def penalty(w, mu):
            return mu * np.linalg.norm(w[:-1]) ** 2 / 2.
    else:
        def penalty(w, mu):
            return mu * np.linalg.norm(w) ** 2 / 2.

    return partial(objective_function, penalty=penalty)


def grad_objective_function_factory(fit_intercept):

    if fit_intercept:
        def grad_penalty(w, mu):
            mu_vec = mu * np.ones(len(w))
            mu_vec[-1] = 0.
            return mu_vec * w
    else:
        def grad_penalty(w, mu):
            return mu * w

    return partial(grad_objective_function, grad_penalty=grad_penalty)
