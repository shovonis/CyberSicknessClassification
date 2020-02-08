import os

import numpy as np

import data_generator as dg
import data_processor_test as dp

# Data Directory and file name
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')
fileName = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

# Get the data from file
raw_data = dp.get_data(fileName)
(header, float_data) = dp.split_data(raw_data)

# Normalize the data
float_data = dg.normalize_data(float_data)

# Define the parameters
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = dg.generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
                         step=step, batch_size=batch_size)

val_gen = dg.generator(float_data,
                       lookback=lookback,
                       delay=delay,
                       min_index=200001,
                       max_index=300000,
                       step=step,
                       batch_size=batch_size)

test_gen = dg.generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=300001,
                        max_index=None,
                        step=step,
                        batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
       # print(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()
