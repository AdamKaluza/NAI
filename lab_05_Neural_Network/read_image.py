"""
==========================================
Classify objects from a photo using the ResNet50 model learned on the imagenet dataset.
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install numpy
pip install tensorflow
pip install tensorflow
==========================================
Usage:
python read_image.py
==========================================
"""

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from IPython.display import Image

# Sample photo for qualification
Image(filename='boxer.jpg')
# Image(filename='york.jpg')

img_path = 'boxer.jpg'
# img_path = 'york.jpg'

# Scaling the photo to 224x224 pix. Image loading.
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Normalize image data before entering it as input.
x = preprocess_input(x)

# Loading the model
model = ResNet50(weights='imagenet')

predictions = model.predict(x)

# Decode the result into a list (class, description, probability), displaying the top 3 results
print('Predicted:', decode_predictions(predictions, top=3)[0])
