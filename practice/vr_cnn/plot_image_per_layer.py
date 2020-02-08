from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load a previously saved model
model = load_model('cats_and_dogs_small_1.h5')
model.summary()

# Directory for a particular image
current_dir = os.path.dirname(__file__)
img_path = os.path.join(current_dir, 'image_manager/cats_and_dogs_small/train/cat.1700.jpg')

# Load one single image
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.00

# Verify and Plot image
plt.imshow(img_tensor[0])
plt.show()
