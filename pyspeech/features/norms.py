import numpy as np


def normalise_by_column(dataset, metric_='minmax'):
    normalise(dataset.T, metric_)


def normalise(dataset, metric='minmax'):
    for j in range(dataset.shape[0]):
        normalised_row = normalise_by(metric, dataset[j])
        dataset[j] = normalised_row


def normalise_by(metric, sample):
    if metric == 'minmax':
        return minmax(sample)
    elif metric == 'stdscore':
        return std_score(sample)
    else:
        print('unknown normalisation')


def std_score(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def minmax(x):
    min_ = x.min()
    max_ = x.max()
    return (x-min_) / (max_-min_)


def mean_normalise(feature):
    feature -= np.mean(feature, axis=0) + 1e-8

