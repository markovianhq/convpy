import os
import pickle

from convpy.estimation import ConversionEstimator
from convpy.preprocessing import OneHotEncoderCOO, FeatureHasher
from convpy.utils import MultiEncoder, MultiEstimator


serialized_attributes = [
    'coef_',
    'lag_coef_',
    'end_time'
]


def to_file(estimator, filename, encoder=None):
    """
    Writes trained estimator and feature encoder/hasher to disc.

    :param estimator: convpy.estimation.ConversionEstimator, The trained estimator.
    :param filename: str, Full path of where estimator is to be written to.
    :param encoder: convpy.preprocessing.OneHotEncoderCOO or convpy.preprocessing.FeatureHasher,
        feature encoder/hasher used.
    :return:
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    serialized_model = get_serialized_model(estimator, encoder=encoder)
    with open(filename, 'wb') as f:
        pickle.dump(serialized_model, f)


def multi_to_file(multi_estimator, filename, multi_encoder):
    """
    Writes trained multi-estimator and feature encoder/hasher to disc.

    :param multi_estimator: MultiEstimator object.
    :param filename: str, Full path of where estimator is to be written to.
    :param multi_encoder: MultiEncoder object.
    :return:
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    serialized_model_list = []
    for k, estimator in enumerate(multi_estimator.estimators):
        serialized_model_list.append(
            get_serialized_model(estimator, encoder=multi_encoder.encoders[k])
        )

    with open(filename, 'wb') as f:
        pickle.dump(serialized_model_list, f)


def get_serialized_model(estimator, encoder=None):

    serialized_model = {a: getattr(estimator, a) for a in serialized_attributes}
    if encoder is not None:
        serialized_model.update({'features': encoder.features, 'add_intercept': encoder.add_intercept})

        type_ = encoder.__class__.__name__
        if type_ == 'OneHotEncoderCOO':
            serialized_model.update({'keymap': encoder.keymap})
        elif type_ == 'FeatureHasher':
            serialized_model.update({'modulo': encoder.modulo})

    return serialized_model


def from_file(filename):
    """
    Loads trained estimator and feature encoder/hasher from disc.

    :param filename: str, Full path to serialized estimator on disc.
    :return: tuple, (convpy.ConversionEstimator, convpy.preprocessing.OneHotEncoderCOO / .FeatureHasher)
    """

    with open(filename, 'rb') as f:
        serialized_model = pickle.load(f)

    estimator, encoder = get_model(serialized_model)

    return estimator, encoder


def multi_from_file(filename):
    """
    Loads trained multi-estimator and feature encoder/hasher from disc.

    :param filename: str, Full path to serialized estimator on disc.
    :return: tuple, (MultiEstimator, MultiEncoder)
    """

    with open(filename, 'rb') as f:
        serialized_model_list = pickle.load(f)

    estimator_list = []
    encoder_list = []
    for serialized_model in serialized_model_list:
        estimator, encoder = get_model(serialized_model)
        estimator_list.append(estimator)
        encoder_list.append(encoder)

    multi_estimator = MultiEstimator(estimators=estimator_list)
    multi_encoder = MultiEncoder(encoders=encoder_list)

    return multi_estimator, multi_encoder


def get_model(serialized_model):

    estimator = ConversionEstimator(end_time=serialized_model['end_time'])
    estimator.coef_ = serialized_model['coef_']
    estimator.lag_coef_ = serialized_model['lag_coef_']

    if serialized_model.get('keymap') is not None:
        encoder = OneHotEncoderCOO(features=serialized_model['features'],
                                   add_intercept=serialized_model['add_intercept']
                                   )
        encoder.keymap = serialized_model['keymap']

    elif serialized_model.get('modulo') is not None:
        encoder = FeatureHasher(serialized_model['modulo'],
                                features=serialized_model['features'],
                                add_intercept=serialized_model['add_intercept']
                                )
    else:
        encoder = None

    return estimator, encoder
