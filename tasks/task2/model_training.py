import numpy as np
from random import shuffle
from tasks.task2.model import save_model


class Params:
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.01
    LABEL_COUNT = 10
    INPUT_SIZE = 784
    SHUFFLE_DATA = True
    BEST_LABELED = 0.0
    BEST_EPOCH = 0
    SAVE_BEST = True


def shuffle_data(features, labels):
    permutation = np.arange(features.shape[0])
    np.random.shuffle(permutation)
    return features[permutation], labels[permutation]


def batch_data(features, labels):
    # Shuffle
    if Params.SHUFFLE_DATA:
        features, labels = shuffle_data(features, labels)

    # Check if there is leftover
    leftover = features.shape[0] % Params.BATCH_SIZE
    if leftover != 0:
        features = features[:-leftover]
        labels = labels[:-leftover]

    # Batch
    batches_count = features.shape[0] // Params.BATCH_SIZE
    batched_features = np.split(features, batches_count)
    batched_labels = np.split(labels, batches_count)

    return zip(batched_features, batched_labels)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def predict(model, input):
    return np.dot(input, model[0]) + model[1]


def format_percentage(amount, total):
    return f"{amount} / {total} ({round(amount / total * 100, 2)}%)"


def validate_model(model, features, labels, epoch_number):
    raw_output = predict(model, features)
    output = np.apply_along_axis(np.argmax, -1, raw_output)
    correctly_labeled = (output == labels).sum()
    print(f"[{epoch_number}] Validation accuracy {format_percentage(correctly_labeled, labels.shape[0])}")

    if Params.BEST_LABELED < correctly_labeled:
        Params.BEST_LABELED = correctly_labeled
        Params.BEST_EPOCH = epoch_number
        print(f"[{epoch_number}] New record! (saving...)")
        save_model(model)

    if abs(Params.BEST_EPOCH - epoch_number) > 5:
        Params.LEARNING_RATE *= 0.9
        print(f"[{epoch_number}] Lowered learning rate")
        Params.BEST_EPOCH = epoch_number


def batch_train_model(model, batch_data):
    features = batch_data[0]
    labels = batch_data[1]
    raw_output = predict(model, features)
    output = np.apply_along_axis(softmax, -1, raw_output)
    diff = labels - output
    batch_diff = np.sum(diff.T, -1)
    model[0] += np.dot(features.T, diff) * Params.LEARNING_RATE
    model[1] += batch_diff * Params.LEARNING_RATE
    return model


def train_model(model, data):
    train_features, train_labels, \
    validation_features, validation_labels = data
    for epoch_number in range(Params.EPOCHS):
        data_batches = batch_data(train_features, train_labels)
        for data_batch in data_batches:
            model = batch_train_model(model, data_batch)
        validate_model(model, validation_features, validation_labels, epoch_number)
    return model


def test_model(model, data):
    features, labels = data
    raw_output = predict(model, features)
    output = np.apply_along_axis(np.argmax, -1, raw_output)
    correctly_labeled = (output == labels).sum()
    print(f"Testing accuracy {format_percentage(correctly_labeled, labels.shape[0])}")
