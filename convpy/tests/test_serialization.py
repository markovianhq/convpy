import os

from convpy.estimation import ConversionEstimator
from convpy.preprocessing import OneHotEncoderCOO
from convpy.serialization import to_file, from_file, multi_to_file, multi_from_file, serialized_attributes
from convpy.utils import MultiEstimator, MultiEncoder


def test_serialize_estimator(tmpdir, estimator_encoder_factory):
    estimator, encoder = estimator_encoder_factory()

    filename = tmpdir.join('estimator.pickle').strpath

    to_file(estimator, filename, encoder=encoder)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0.

    deserialized_estimator, deserialized_encoder = from_file(filename)

    assert isinstance(deserialized_estimator, ConversionEstimator)
    assert isinstance(deserialized_encoder, OneHotEncoderCOO)

    for a in serialized_attributes:
        assert getattr(estimator, a) == getattr(deserialized_estimator, a)


def test_serialize_multi_estimator(tmpdir, estimator_encoder_factory):
    estimator_list = []
    encoder_list = []
    for _ in [0, 1]:
        estimator, encoder = estimator_encoder_factory()
        estimator_list.append(estimator)
        encoder_list.append(encoder)

    multi_estimator = MultiEstimator(estimator_list)
    multi_encoder = MultiEncoder(encoder_list)
    filename = tmpdir.join('estimator.pickle').strpath

    multi_to_file(multi_estimator, filename, multi_encoder)

    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0.

    deserialized_multi_estimator, _ = multi_from_file(filename)

    assert isinstance(deserialized_multi_estimator, MultiEstimator)

    for k in [0, 1]:
        for a in serialized_attributes:
            assert getattr(multi_estimator.estimators[k], a) == getattr(
                deserialized_multi_estimator.estimators[k], a)
