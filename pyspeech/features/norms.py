import numpy as np


def norm(x, _min, _max):
    return (x-_min) / (_max-_min)


def normalise(dataset):
    for j in range(dataset.shape[0]):
        row_min = dataset[j].min()
        row_max = dataset[j].max()
        normalised_row = [norm(f_i, _min=row_min, _max=row_max) for f_i in dataset[j]]
        dataset[j] = np.array(normalised_row)


def normalise_by_column(dataset):
   normalise(dataset.T) 

