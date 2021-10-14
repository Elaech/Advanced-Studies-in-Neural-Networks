import numpy as np

PATH = "task2_model"


def init_model(input_count, output_count):
    return [np.random.randn(input_count, output_count), np.random.randn(1, output_count)]


def save_model(model):
    np.savez_compressed(PATH, features=model[0], labels=model[1])


def load_model():
    load = np.load(f"{PATH}.npz")
    return [load['features'], load['labels']]
