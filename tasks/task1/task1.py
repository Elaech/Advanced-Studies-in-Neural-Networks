import numpy
import numpy as np
import json
from tasks.task1.data_generation import generate_data, generate_point
from pathlib import Path
from matplotlib import pyplot as plt


def save_model(path, model):
    with open(path, "w+") as output:
        json.dump(model.tolist(), output)


def load_model(path):
    with open(path, "r") as input:
        return np.array(json.load(input))


def generate_data_batch(batch_size):
    p_features, p_labels = generate_data(batch_size // 2, [[0.0, 45.0], [0.0, 100.0], [1.0, 1.0]], 3, -1)
    n_features, n_labels = generate_data(batch_size // 2, [[55.0, 100.0], [0.0, 100.0], [1.0, 1.0]], 3, +1)
    features = np.concatenate([p_features, n_features], axis=0)
    labels = np.concatenate([p_labels, n_labels], axis=0)
    return features, labels


def init_model():
    model = numpy.zeros(shape=3)
    for index in range(3):
        model[index] = generate_point([[-100.0, 100.0]], 1)
    return model


def assign_label(value):
    if value < 0:
        return -1.0
    return 1.0


def model_predict(model, features):
    raw_predictions = np.dot(features, model)
    f = np.vectorize(assign_label)
    return f(raw_predictions)


def fitness(predictions, labels):
    """
    Compute the score based by calculating the difference between the predictions and labels
    """
    return np.sum(
        np.abs(
            (predictions - labels) / 2
        )
    )


def get_vicinity(size=3, start_point=10):
    vicinity = np.zeros(shape=(size * 2,))
    for index in range(0, size * 2, 2):
        step = start_point * (10 ** (-index // 2))
        vicinity[index] = step
        vicinity[1 + index] = -step
    return vicinity


def train_model(model, features, labels, vicinity):
    best_global_model = model
    best_global_score = fitness(model_predict(model, features), labels)

    best_model = best_global_model.copy()
    best_score = best_global_score

    while True:
        for x in vicinity:
            for y in vicinity:
                for z in vicinity:
                    modifications = np.array([x, y, z])
                    new_model = best_global_model + modifications
                    new_score = fitness(model_predict(new_model, features), labels)

                    if new_score > best_score:
                        best_model = new_model
                        best_score = new_score

        if best_global_score == best_score:
            print("At top of the hill, stop searching!")
            break
        else:
            best_global_model = best_model
            best_global_score = best_score
            print(f"Found new best: {best_score}")
    return best_global_model


def format_percentage(amount, total):
    return f"{amount} / {total} ({amount * 100 / total}%)"


def test_model(model, nr_of_tests, batch_size):
    test_scores = np.zeros(shape=(nr_of_tests, 1))
    for index in range(nr_of_tests):
        features, labels = generate_data_batch(batch_size)
        predictions = model_predict(model, features)
        test_scores[index] = fitness(predictions, labels)

    print(f"On average from {nr_of_tests} tests,",
          f" the model correctly classifies {format_percentage(np.average(test_scores), batch_size)}")


def plot_training_result(model, features, batch_size):
    halfpoint = batch_size // 2
    plt.figure(figsize=(15, 8))
    plt.scatter(features[:halfpoint][:, 0], features[:halfpoint][:, 1], c="green")
    plt.scatter(features[halfpoint:][:, 0], features[halfpoint:][:, 1], c="red")

    line_x = np.linspace(0, 100)
    line_y = (-model[0] * line_x - model[2]) / model[1]
    plt.plot(line_x, line_y, c="black")

    plt.show()


def main():
    TRAIN = True
    PLOT = True
    BATCH_SIZE = 1000
    TESTS = 100
    VICINITY_SIZE = 12
    VICINITY_START_POINT = 10e+5

    model_path = Path("task1_model.json")

    if TRAIN:
        model = init_model()
        features, labels = generate_data_batch(BATCH_SIZE)
        vicinity = get_vicinity(VICINITY_SIZE, VICINITY_START_POINT)
        print(f"The vicinity is \n{vicinity}")
        model = train_model(model, features, labels, vicinity)
        save_model(model_path, model)
        if PLOT:
            plot_training_result(model, features, BATCH_SIZE)
    else:
        model = load_model(model_path)
        test_model(model, nr_of_tests=TESTS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
