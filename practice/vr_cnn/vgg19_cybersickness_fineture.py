from time import time
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG19
from keras.callbacks import TensorBoard
import data_source_manager as dm
import plot_network as plot


# Base VGG16 model
conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# We will unfreeze the last layers and freeze all the other layers
set_trainable = True
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        conv_base.trainable = True
    else:
        layer.trainable = False

# Defining the model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# Model Summary
model.summary()
# Compiling the model
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

# Defining the logs for tensor board.
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Run graph
history = model.fit_generator(dm.get_train_data(), steps_per_epoch=100, epochs=20, callbacks=[tensorboard],
                              validation_data=dm.get_validation_data(), validation_steps=50)

model.save('cybersickness_vgg19.h5')

# Plotting the graph
plot.plot_accuracy_and_loss(history)
plot.plot_smothed_acc_loss(history)
