# from keras import Sequential
# from keras.layers import Embedding, SimpleRNN
# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras.layers import Dense
# import matplotlib.pyplot as plt
#
#
# max_features = 10000
# maxlen = 500
# batch_size = 32
#
# print('Loading data...')
# (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(input_train), 'train sequences')
# print(len(input_test), 'test sequences')
# print('Pad sequences (samples x time)')
# input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
# input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
# print('input_train shape:', input_train.shape)
# print('input_test shape:', input_test.shape)
#
# model = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(SimpleRNN(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(input_train, y_train, epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)
#
# ###################### Plotting ###########################
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# a = 9
# b = 4
# div2 = a // b
# print(div2)

import numpy as np

# samples = np.zeros((128, 120, 3))
# indices = range(0, 720, 6)
# for n in indices:
#   print(n)
data = np.random.randint(20000, size=(20000, 3))
rows = np.arange(720, 848)
print("TEST:: ", data[range(0, 720, 6)])
# print(data.shape[-1])
samples = np.zeros((len(rows), 720 // 6, data.shape[-1]))
for j, row in enumerate(rows):
    print("Current: ", j, row)
    indices = range(rows[j] - 720, rows[j], 6)
    print("Indices: ", indices)
    samples[j] = data[indices]
    print("Samples Cur: ", samples[j])


