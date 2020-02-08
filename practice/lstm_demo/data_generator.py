import numpy as np


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1  # set max index to the highest index possible.

    minimum_index_bound = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if minimum_index_bound + batch_size >= max_index:
                minimum_index_bound = min_index + lookback
            rows = np.arange(minimum_index_bound, min(minimum_index_bound + batch_size, max_index))
            minimum_index_bound += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


def normalize_data(data, min_value=0, max_value=None):

    if max_value is None:
        max_value = len(data) - 1

    mean = data[min_value:max_value].mean(axis=0)
    data -= mean
    std = data[min_value:max_value].std(axis=0)
    data /= std

    return data
