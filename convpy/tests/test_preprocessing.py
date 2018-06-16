from random import random
import numpy as np

from convpy.preprocessing import FeatureHasher, feature_hash_transform, feature_hash, OneHotEncoderCOO


def test_hash_function():
    for _ in range(50):
        modulo = 1 + int(10 * random())

        x = np.array([random() for _ in range(100)])
        x = feature_hash(x, modulo)

        assert len(x) == modulo


def test_feature_hash_transform():
    x = np.array([['a', 'A'], ['b', 'A'], ['c', 'B'], ['a', 'B']])

    def transform(X, mod):
        outdata = []
        for row in X:
            outdata.append(feature_hash(row, mod))

        return np.asarray(outdata)

    for i in range(2, 12):
        assert np.linalg.norm(transform(x, i) - feature_hash_transform(x, i).toarray()) == 0.


def test_onehotencoding():
    x = np.array([['a', 'A'], ['b', 'A'], [np.nan, 'B'], ['a', 'B']], dtype=np.object)
    encoder = OneHotEncoderCOO()

    x_s = encoder.transform(x)
    set_ = set()

    for i in range(2):
        set_ = set_.union(set(encoder.keymap[i].keys()))

    assert len(set_.symmetric_difference({'a', 'b', np.nan, 'A', 'B'})) == 0
    assert x_s.shape == (4, 5)


def test_onehotencoding_with_intercept():
    x = np.array([['a', 'A'], ['b', 'A'], [np.nan, 'B'], ['a', 'B']], dtype=np.object)
    encoder = OneHotEncoderCOO(add_intercept=True)

    x_s = encoder.transform(x)
    set_ = set()

    for i in range(2):
        set_ = set_.union(set(encoder.keymap[i].keys()))

    assert len(set_.symmetric_difference({'a', 'b', np.nan, 'A', 'B'})) == 0
    assert x_s.shape == (4, 6)
    assert all(x_s.toarray()[:, -1] == 1.)


def test_onehotencoding_with_features():
    x = np.array([['a', 'A'], ['b', 'A'], [np.nan, 'B'], ['a', 'B']], dtype=np.object)
    encoder = OneHotEncoderCOO(features=['feature_1', 'feature_2'], add_intercept=True)

    encoder.transform(x)
    dict_ = {'feature_1': 'a', 'feature_2': 'A'}
    assert np.linalg.norm((encoder.dict_vectorizer(dict_) - encoder.transform(np.array([['a', 'A']]))).toarray()) == 0.


def test_featurehasher():
    x = np.array([['a', 'A'], ['b', 'A'], ['c', 'B'], ['a', 'B']])
    mod = 1 + int(10 * random())
    hasher = FeatureHasher(modulo=mod)

    def transform(X, mod):
        outdata = []
        for row in X:
            outdata.append(feature_hash(row, mod))

        return np.asarray(outdata)

    assert np.linalg.norm(transform(x, mod) - hasher.transform(x).toarray()) == 0.

    hasher.features = ['feature_1', 'feature_2']
    test_vector = hasher.dict_vectorizer({'feature_1': 'a', 'feature_2': 'B'}).toarray()

    assert np.linalg.norm(transform([['a', 'B']], mod) - test_vector) == 0.


def test_featurehasher_intercept():
    x = np.array([['a', 'A'], ['b', 'A'], ['c', 'B'], ['a', 'B']])
    mod = 1 + int(10 * random())
    hasher = FeatureHasher(modulo=mod, add_intercept=True)

    def transform(X, mod, add_intercept=True):
        outdata = []
        for row in X:
            outdata.append(feature_hash(row, mod, add_intercept=add_intercept))

        return np.asarray(outdata)

    assert np.linalg.norm(transform(x, mod, add_intercept=True) - hasher.transform(x).toarray()) == 0.
    assert np.linalg.norm(hasher.transform(x).toarray()[:, -1] - 1) == 0.

    hasher.features = ['feature_1', 'feature_2']
    test_vector = hasher.dict_vectorizer({'feature_1': 'a', 'feature_2': 'B'}).toarray()

    assert np.linalg.norm(transform([['a', 'B']], mod) - test_vector) == 0.
