import numpy as np


def apply_at_col(dataset, metric_='minmax'):
    normalise(dataset.T, metric_)


def apply(dataset, metric='minmax'):
    for j in range(dataset.shape[0]):
        normalised_row = normalise_by(metric, dataset[j])
        dataset[j] = normalised_row


def _normalise_by(metric, sample):
    if metric == 'minmax':
        return _minmax(sample)
    elif metric == 'stdscore':
        return _std_score(sample)
    else:
        print('unknown normalisation')


def _minmax(x):
    min_ = x.min()
    max_ = x.max()
    return (x-min_) / (max_-min_)


def _std_score(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def mean_normalise(feature):
    feature -= np.mean(feature, axis=0) + 1e-8

