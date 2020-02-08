import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('data/raw_data.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
n_hours = 60
n_features = 13
reframed = series_to_supervised(scaled, n_hours, 1)
reframed.to_csv('reframe.csv')

# split into train and test sets
values = reframed.values
n_train_hours = 20000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(64,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.summary()
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=128,
                    validation_data=(test_X, test_y), verbose=1,
                    shuffle=False)

# # design network
# model = Sequential()
# model.add(LSTM(32,input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# model.summary()
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=128,
#                     validation_data=(train_X, train_y), verbose=2,
#                     shuffle=False)
#
# plot history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b-o', label='Training loss')
plt.plot(epochs, val_loss, 'r-*', label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(min(epochs), max(epochs) + 1, 5.0))
plt.legend()
plt.savefig('test.png')
plt.show()

# Evaluate the model
test_loss = model.evaluate(test_X, test_y, verbose=1)
print("Test Loss", test_loss)
predictions = model.predict(test_X, verbose=0)
np.savetxt("predictions.csv", predictions * 10, delimiter=",")
np.savetxt("actuals.csv", test_y, delimiter=",")

# serialize weights to HDF5
model.save_weights("lstm_32_128b_50epoch.h5")
print("Saved model to disk")
