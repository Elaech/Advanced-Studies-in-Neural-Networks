import pickle
import gzip
import numpy as np
from tasks.task2.model_training import Params


def load_mnist_data():
    with gzip.open('mnist.pkl.gz', 'rb') as input_zip:
        return pickle.load(input_zip, encoding='latin')


def split_features_labels(data):
    features = np.array(data[0])
    labels = np.array(data[1])
    return features, labels


def get_train_data():
    train_data, validation_data, test_data = load_mnist_data()
    train_features, train_labels = split_features_labels(train_data)
    validation_features, validation_labels = split_features_labels(validation_data)
    return train_features, mass_one_hot(train_labels, Params.LABEL_COUNT), \
           validation_features, validation_labels


def get_test_data():
    train_data, validation_data, test_data = load_mnist_data()
    test_features, test_labels = split_features_labels(test_data)
    return test_features, test_labels


def mass_one_hot(array, size):
    return np.array([one_hot(x, size) for x in array])


def one_hot(x, size):
    hot = np.zeros(size, dtype=float)
    hot[x] = 1.0
    return hot
