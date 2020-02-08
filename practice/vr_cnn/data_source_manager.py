import os
from keras.preprocessing.image import ImageDataGenerator

# Defining data source directory
current_dir = os.path.dirname(__file__)
train_dir = os.path.join(current_dir, 'datasource/train/')
validation_dir = os.path.join(current_dir, 'datasource/test/')
test_dir = os.path.join(current_dir, 'datasource/test/')



# Image Generator for getting the image information.
def get_train_data():
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, )

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=20,
                                                        class_mode='binary')
    return train_generator


def get_test_data():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')
    return test_generator


def get_validation_data():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=20,
                                                            class_mode='binary')
    return validation_generator
