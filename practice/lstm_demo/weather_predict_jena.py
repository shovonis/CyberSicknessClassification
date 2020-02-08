import os
import numpy as np

# Data Directory and file name
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data')
fname = os.path.join(data_dir, 'dataset.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# Float Data from file
float_data = np.zeros((len(lines), len(header) - 1))

print(float_data.shape)
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

print(float_data)
# Normalize the data
# mean = float_data[:200000].mean(axis=0)
# float_data -= mean
# std = float_data[:200000].std(axis=0)
# float_data /= std



#
# # Generator to generate the data
# def generator(data, lookback, delay, min_index, max_index,
#               shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     print("i is: ", i)
#     while 1:
#         if shuffle:
#             rows = np.random.randint(
#                 min_index + lookback, max_index, size=batch_size)
#             print("Rows: ", rows)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             print("ROWS: ", list(rows))
#             i += len(rows)
#             print("i is: ", i)
#
#         samples = np.zeros((len(rows),
#                             lookback // step,
#                             data.shape[-1]))
#         print("Samples: ", samples)
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             print("Samples at J: ", samples[j])
#             targets[j] = data[rows[j] + delay][1]
#             print("Target at J: ", targets[j])
#
#         print("Full Sample: ", samples)
#         print("Full Target: ", targets)
#         yield samples, targets
#
#
# # Parameters
# lookback = 1440
# step = 6
# delay = 144
# batch_size = 128
#
# train_gen = generator(float_data,
#                       lookback=lookback,
#                       delay=delay,
#                       min_index=0,
#                       max_index=200000,
#                       shuffle=False,
#                       step=step,
#                       batch_size=batch_size)
#
# # This is how many steps to draw from `val_gen`
# # in order to see the whole validation set:
# val_steps = (300000 - 200001 - lookback) // batch_size
#
# # This is how many steps to draw from `test_gen`
# # in order to see the whole test set:
# test_steps = (len(float_data) - 300001 - lookback) // batch_size
#
