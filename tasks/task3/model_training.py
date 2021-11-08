import numpy as np
from tasks.task3.model import save_model


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def switch_dropout(value: bool) -> bool:
    old = Params.DROPOUT
    Params.DROPOUT = value
    return old


class Params:
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY = 0.9
    ITERATION_CHECK = 20
    ITERATION_CHECK_DECAY = 1
    LABEL_COUNT = 10
    INPUT_SIZE = 784
    HIDDEN_LAYERS = [100]
    DROP_RATE = [0.2, 0.0]
    DROPOUT = True
    NOISE = {
        "mean": 0,
        "stddev": 0.1
    }
    ACTIVATION = [sigmoid, softmax]
    BACKWARDS_ACTIVATION = sigmoid_derivative
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


def format_percentage(amount, total):
    return f"{amount} / {total} ({round(amount / total * 100, 2)}%)"


def one_layer_predict(layer, input):
    return np.dot(input, layer[0]) + layer[1]


def get_dropout(model):
    if not Params.DROPOUT:
        return [1] * len(model)
    dropout = []
    for index, layer in enumerate(model):
        drop_rate = Params.DROP_RATE[index]
        aug_rate = 1 / (1 - drop_rate)
        dropout.append(
            np.random.choice([0, aug_rate], layer[0].shape[1], p=[drop_rate, 1 - drop_rate])
        )
    return dropout


def feed_forward(model, features):
    outputs = [features, ]
    dropout = get_dropout(model)
    activation = Params.ACTIVATION
    for index, layer in enumerate(model):
        layer_output = np.apply_along_axis(activation[index],
                                           axis=-1,
                                           arr=one_layer_predict(layer, outputs[len(outputs) - 1])
                                           )
        layer_output = np.multiply(layer_output, dropout[index])
        outputs.append(layer_output)
    return outputs


def compute_errors(model, output, labels):
    # softmax derivative for one hot
    L_errors = output[len(output) - 1] - labels
    errors = [L_errors]
    for index in range(len(output) - 2, -1, -1):
        next_layer_errors = errors[len(errors) - 1]
        next_layer = model[index + 1]
        layer_output = output[index]
        layer_errors = Params.BACKWARDS_ACTIVATION(layer_output) * np.dot(next_layer_errors, next_layer[0].T)
        errors.append(layer_errors)
    return errors[::-1]


def compute_updates(output, errors):
    updates = []
    for index, layer_errors in enumerate(errors):
        weight_update = Params.LEARNING_RATE * np.dot(output[index].T, layer_errors)
        bias_update = Params.LEARNING_RATE * np.sum(layer_errors)
        updates.append([weight_update, bias_update])
    return updates


def back_propagate(model, output, labels):
    errors = compute_errors(model, output[1:], labels)
    updates = compute_updates(output, errors)
    for index, layer in enumerate(model):
        layer[0] = layer[0] - updates[index][0]
        layer[1] = layer[1] - updates[index][1]
    return model


def batch_train_model(model, batch_data):
    features = batch_data[0]
    labels = batch_data[1]
    output = feed_forward(model, features)
    model = back_propagate(model, output, labels)
    return model


def augment_features(features):
    noise = np.random.normal(Params.NOISE['mean'], Params.NOISE['stddev'], size=features.shape)
    noise = np.absolute(noise)
    return features + noise


def train_model(model, data):
    train_features, train_labels, \
    validation_features, validation_labels = data
    for epoch_number in range(Params.EPOCHS):
        augmented_features = augment_features(train_features)
        data_batches = batch_data(augmented_features, train_labels)
        for data_batch in data_batches:
            model = batch_train_model(model, data_batch)
        validate_model(model, validation_features, validation_labels, epoch_number)
    return model


def predict(model, input):
    output = feed_forward(model, input)
    return output[len(output) - 1]


def validate_model(model, features, labels, epoch_number):
    old_state = switch_dropout(False)
    raw_output = predict(model, features)
    output = np.apply_along_axis(np.argmax, -1, raw_output)
    correctly_labeled = (output == labels).sum()
    print(f"[{epoch_number}] Validation accuracy {format_percentage(correctly_labeled, labels.shape[0])}")

    if Params.BEST_LABELED < correctly_labeled:
        Params.BEST_LABELED = correctly_labeled
        Params.BEST_EPOCH = epoch_number
        print(f"[{epoch_number}] New record! (saving...)")
        save_model(model)

    if abs(Params.BEST_EPOCH - epoch_number) > Params.ITERATION_CHECK:
        Params.LEARNING_RATE *= Params.LEARNING_RATE_DECAY
        print(f"[{epoch_number}] Lowered learning rate")
        Params.BEST_EPOCH = epoch_number
    Params.ITERATION_CHECK *= Params.ITERATION_CHECK_DECAY
    switch_dropout(old_state)


def test_model(model, data):
    old_state = switch_dropout(False)
    features, labels = data
    raw_output = predict(model, features)
    output = np.apply_along_axis(np.argmax, -1, raw_output)
    correctly_labeled = (output == labels).sum()
    print(f"Testing accuracy {format_percentage(correctly_labeled, labels.shape[0])}")
    switch_dropout(old_state)
