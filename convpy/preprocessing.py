from random import random
import mmh3
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack


def feature_hash_transform(X, modulo):
    matdata = [1. for _ in range(X.shape[0] * X.shape[1])]  # assumes (n, p)-dimensional array X
    matrow = []
    matcol = []
    for i, row in enumerate(X):
        matrow += [i for _ in range(X.shape[1])]
        matcol += [mmh3.hash(feature) % modulo for feature in row]

    return csr_matrix((matdata, (matrow, matcol)), shape=(X.shape[0], modulo))


def feature_hash(x, mod, add_intercept=False):
    x_h = np.array(mod * [0] + add_intercept * [1])
    for i in x:
        x_h[mmh3.hash(i) % mod] += 1
    return x_h


def signed_feature_hash(x, mod):
    x_h = np.zeros(mod)
    for i in x:
        x_h[mmh3.hash(i) % mod] += 1 - 2 * (hash(i) % 2)  # export PYTHONHASHSEED="0" to make hash function non-random
    return x_h


class FeatureHasher:

    def __init__(self, modulo, features=None, add_intercept=False):
        self.modulo = modulo
        self.features = features
        self.add_intercept = add_intercept

    def transform(self, X):
        bool_ = self.add_intercept
        n, p = X.shape
        matdata = [1. for _ in range(n * (p + bool_))]  # assumes (n, p)-dimensional array X
        matrow = []
        matcol = []
        for i, row in enumerate(X):
            matrow += (p + bool_) * [i]
            matcol += [mmh3.hash(feature) % self.modulo for feature in row] + bool_ * [self.modulo]

        return csr_matrix((matdata, (matrow, matcol)), shape=(n, self.modulo + bool_))

    def dict_vectorizer(self, dict_):
        bool_ = self.add_intercept
        p = len(self.features)
        matdata = (p + bool_) * [1.]
        matrow = (p + bool_) * [0]
        # if feature not present in dict_ choose a random string
        matcol = [mmh3.hash(self.formatter(f, dict_.get(f, str(random()))))
                  % self.modulo for f in self.features] + bool_ * [self.modulo]

        return csr_matrix((matdata, (matrow, matcol)), shape=(1, self.modulo + bool_))

    @staticmethod
    def formatter(feature, value):
        return str(value) if value is not None else '{feature}_na'.format(feature=feature)


class OneHotEncoderCOO:
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix doing one-of-k encoding.
    Parts of code borrowed from Paul Duan
    (LICENSE: https://github.com/pyduan/amazonaccess/blob/master/MIT-LICENSE),
    Mahendra Kariya, Christophe Bourguignat.
    """

    def __init__(self, features=None, add_intercept=False):
        self.keymap = None
        self.features = features
        self.add_intercept = add_intercept

    def fit(self, x):
        self.keymap = []
        for col in x.T:
            uniques = set(list(col))
            self.keymap.append({key: i for i, key in enumerate(uniques)})

    def partial_fit(self, x):
        """
        This method can be used for doing one hot encoding in mini-batch mode.
        """
        if self.keymap is None:
            self.fit(x)
        else:
            for i, col in enumerate(x.T):
                uniques = set(self.keymap[i].keys()).union(list(col))
                self.keymap[i] = {key: i for i, key in enumerate(uniques)}

    def transform(self, x):
        if self.keymap is None:
            self.fit(x)

        n = x.shape[0]
        outdat = []
        for i, col in enumerate(x.T):
            matrow = []
            matcol = []
            matdata = []
            km = self.keymap[i]
            num_labels = len(km)
            for j, val in enumerate(col):
                if val in km:
                    matrow.append(j)
                    matcol.append(km[val])
                    matdata.append(1)
            spmat = coo_matrix((matdata, (matrow, matcol)), shape=(n, num_labels))
            outdat.append(spmat)

        if self.add_intercept:
            intercept_column = coo_matrix(([1. for _ in range(n)], ([i for i in range(n)], [0 for _ in range(n)])),
                                          shape=(n, 1))
            outdat.append(intercept_column)

        outdat = hstack(outdat).tocsr()

        return outdat

    def dict_vectorizer(self, dict_):
        try:
            x = np.array([[self.formatter(dict_.get(f, '')) for f in self.features]])
        except TypeError:
            raise AttributeError('Object has no attribute "feature".'
                                 'Set this attribute before using the dict_vectorizer method.')
        return self.transform(x)

    @staticmethod
    def formatter(value):
        return str(value) if value is not None else ''
