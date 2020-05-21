import numpy as np


def apply_at_col(dataset, metric_='minmax'):
    data = np.array(dataset)
    normalised_data = np.empty(data.shape)
    for j, col in enumerate(data.T):
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
    return (x-min_) / (max_-min_ + 1e-15)


def _std_score(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def avg_reduction(feature):
    normalised = feature - np.mean(feature, axis=0)
    return normalised


def mean_normalise(feature):
    f_less_avg = avg_reduction(feature)
    minmax = np.max(feature, axis=0) - np.min(feature, axis=0)
    return f_less_avg / (minmax + 1e-15)

