import numpy as np


def apply_at_col(dataset, metric_='minmax'):
    normalised_data = np.empty(dataset.shape)
    for j, col in enumerate(dataset.T):
        normalised_data[:, j] = _normalise_by(metric_, col)
    return normalised_data


def apply(dataset, metric='minmax'):
    for j in range(dataset.shape[0]):
        normalised_row = _normalise_by(metric, dataset[j])
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


def average_reduction(feature):
    normalised = feature - np.mean(feature, axis=0) + 1e-8
    return normalised

def mean_normalise(feature):
    f_less_avg = average_reduction(feature)
    minmax = np.max(feature, axis=0) - np.min(features, axis=0)
    return f_less_avg / minmax

