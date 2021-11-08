import numpy as np

PATH = "task3_model_test"


def init_model(input_count, hidden_layers, output_count):
    model = []
    connectivity = [input_count] + hidden_layers + [output_count]
    for index in range(len(connectivity) - 1):
        stddev = 1 / np.sqrt(connectivity[index])
        weights = stddev * np.random.randn(connectivity[index], connectivity[index + 1])
        biases = np.random.randn(connectivity[index + 1])
        model.append([weights, biases])
    return model


def save_model(model):
    np.savez_compressed(PATH, l1w=model[0][0], l1b=model[0][1], l2w=model[1][0], l2b=model[1][1])


def load_model():
    load = np.load(f"{PATH}.npz")
    return [[load['l1w'], load['l1b']], [load['l2w'], load['l2b']]]
