import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm


def prepare_input(y, X, end_time):

    y0, y1 = y[np.isnan(y[:, 1])], y[~np.isnan(y[:, 1])]
    x0, x1 = X[np.isnan(y[:, 1])], X[~np.isnan(y[:, 1])]

    diagonal0, diagonal1 = coo_matrix((y0.shape[0], y0.shape[0])), coo_matrix((y1.shape[0], y1.shape[0]))
    diagonal0.setdiag(np.ones(y0.shape[0]))
    diagonal1.setdiag(np.ones(y1.shape[0]))

    mu = get_regularization_parameter(X)

    return {'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1, 'end_time': end_time, 'mu': mu,
            'diagonal0': diagonal0, 'diagonal1': diagonal1}


def get_regularization_parameter(X):
    n = X.shape[0]
    return norm(X) ** 2 / n


def hash_all(x, mod):
    x_ = np.zeros(mod)
    for i in x:
        x_[hash(i) % mod] += 1
    return x_


def check_input_data(y):
    assert (y[:, 0] >= 0.).all()
    assert (y[~np.isnan(y[:, 1])][:, 0] <= y[~np.isnan(y[:, 1])][:, 1]).all()


class MultiEncoder:
    def __init__(self, encoders):
        """
        :param encoders: iterable of encoders with the property:
            encoders[i].features is a subset of encoders[i+1].features
        """
        self.encoders = encoders
        self.dimension = len(encoders)

    def dict_vectorizer(self, state):
        num_common_feat = len(set(self.encoders[-1].features).intersection(state))
        best_level, best_encoder = self.dimension, self.encoders[-1]

        for level, encoder in reversed(list(enumerate(self.encoders))):
            partial_features = set(encoder.features)
            num_common_feat_level = len(partial_features.intersection(state))
            if num_common_feat_level < num_common_feat:
                break
            else:
                best_level, best_encoder = level, encoder

        return best_level, best_encoder.dict_vectorizer(state)


class MultiEstimator:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, x_):
        level, x = x_
        estimator = self.estimators[level]
        return estimator.predict(x)
