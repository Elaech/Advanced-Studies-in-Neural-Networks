import random as rd
import numpy as np


def generate_point(constraints, nr_of_dimensions):
    point = np.empty(shape=(nr_of_dimensions,), dtype=float)
    for index in range(nr_of_dimensions):
        point[index] = (rd.uniform(constraints[index][0], constraints[index][1]))
    return point


def generate_point_list(points_count, constraints, nr_of_dimensions):
    point_list = np.zeros(shape=(points_count, nr_of_dimensions), dtype=float, )
    for index in range(points_count):
        point_list[index] = generate_point(constraints, nr_of_dimensions)
    return point_list


def generate_data(data_count, constraints, nr_of_dimensions, label):
    features = generate_point_list(data_count, constraints, nr_of_dimensions)
    labels = np.full(shape=(data_count,), fill_value=label)
    return features, labels
