import numpy as np
from scipy.special import expit as sigmoid
# formulas here hold for more general functions


def d_sigmoid(t):
    return sigmoid(t) * (1 - sigmoid(t))


def hazard(x, w_d):
    return np.exp(np.dot(x, w_d))


def grad_hazard(x, w_d):
    return (x.T * hazard(x, w_d)).T


def hidden(x, w_c):
    return sigmoid(np.dot(x, w_c))


def grad_hidden(x, w_c):
    return (x.T * d_sigmoid(np.dot(x, w_c))).T


def log_likelikood_conversion(x, w_c, w_d, time_delta):
    return np.log(hidden(x, w_c)) + np.log(hazard(x, w_d)) - hazard(x, w_d) * time_delta


def grad_w_c_log_likelikood_conversion(x, w_c):
    return (grad_hidden(x, w_c).T / hidden(x, w_c)).T


def grad_w_d_log_likelikood_conversion(x, w_d, time_delta):
    return (grad_hazard(x, w_d).T * (1 / hazard(x, w_d) - time_delta)).T


def log_likelikood_non_conversion(x, w_c, w_d, elapsed_time):
    return np.log(1 - hidden(x, w_c) + hidden(x, w_c) * np.exp(-hazard(x, w_d) * elapsed_time))


def grad_w_c_log_likelikood_non_conversion(x, w_c, w_d, elapsed_time):
    norm = 1 - hidden(x, w_c) + hidden(x, w_c) * np.exp(-hazard(x, w_d) * elapsed_time)

    return (grad_hidden(x, w_c).T * (- 1 + np.exp(-hazard(x, w_d) * elapsed_time)) / norm).T


def grad_w_d_log_likelikood_non_conversion(x, w_c, w_d, elapsed_time):
    temp = hidden(x, w_c) * np.exp(-hazard(x, w_d) * elapsed_time)
    norm = 1 - hidden(x, w_c) + temp

    inner = - grad_hazard(x, w_d).T * elapsed_time * temp

    return (inner / norm).T


def objective_function(w, Jacobian, df0, df1, x0, x1, click_time, conv_time, end_time):
    # neg_log_likelihood
    w_c, w_d = np.split(w, 2)

    a0 = log_likelikood_non_conversion(x0, w_c, w_d, end_time - df0[click_time])
    T = df1[conv_time] - df1[click_time]
    a1 = log_likelikood_conversion(x1, w_c, w_d, T)

    return - np.sum(a0) - np.sum(a1)


def grad_objective_function(w, Jacobian, df0, df1, x0, x1, click_time, conv_time, end_time):
    w_c, w_d = np.split(w, 2)

    p = x0.shape[1]
    n0 = df0.shape[0]
    n1 = df1.shape[0]

    g0c = grad_w_c_log_likelikood_non_conversion(x0, w_c, w_d, end_time - df0[click_time])
    g1c = grad_w_c_log_likelikood_conversion(x1, w_c)

    g0d = grad_w_d_log_likelikood_non_conversion(x0, w_c, w_d, end_time - df0[click_time])
    g1d = grad_w_d_log_likelikood_conversion(x1, w_d, df1[conv_time] - df1[click_time])

    Jacobian[:p] = - np.dot(np.ones(n0), g0c) - np.dot(np.ones(n1), g1c)
    Jacobian[p:] = - np.dot(np.ones(n0), g0d) - np.dot(np.ones(n1), g1d)

    return Jacobian
