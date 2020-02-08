from time import time
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG19
from keras.callbacks import TensorBoard
import data_source_manager as dm
import plot_network as plot
from keras.models import load_model
import numpy as np


model = load_model('cybersickness_vgg19.h5')
model.load_weights('cybersickness_vgg19.h5')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

data = dm.get_test_data()
test_loss, test_acc = model.evaluate_generator(data, steps=50)
print('test acc:', test_acc)
print('test acc:', test_loss)


k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)


model.train(data)
test_loss, test_acc = model.evaluate_generator(data, steps=50)
print('test acc:', test_acc)
print('test acc:', test_loss)