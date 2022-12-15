"""
==========================================
Teaching a neural network to classify data from MNIST dataset
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install numpy
pip install matplotlib
pip install tensorflow
==========================================
Usage:
python fashion.py
==========================================
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Fashon MINST collection from the Tensoflow library
from tensorflow.python.ops.confusion_matrix import confusion_matrix

fashion_mnist = tf.keras.datasets.fashion_mnist

# Assigning data to sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Each image is mapped to a single label. Since class names are not included in the data set,
# Save them here for later use when printing images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale the values to a range from 0 to 1 and divide by 255 before feeding them into the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# Displaying the first 25 images to check if the data is in the right format and that the network is ready for teaching
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Model building. The layers of the neural network extract a representation of the data fed into them.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the model, inputting training data, teaching the model to associate images and labels in 10 epocs.
model.fit(train_images, train_labels, epochs=10)

# We check whether the trained model agrees with the test one
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Once the model is trained, it can be used to predict some images.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# Model provided a label for each image in the test set
# predictions[0]

# The predicted array is 10 numbers represents the property of matching the appropriate label to the image
np.argmax(predictions[0])

# test_labels[0]


def plot_image(i, predictions_array, true_label, img):
    # Creating a chart
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    # Checking forecasts after training the model. Correct forecast labels are blue,and incorrect forecast labels are red
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Once the model is trained, it can be used to forecast some images. Correct forecast labels are blue, and incorrect forecast labels are red.
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Single image forecast

# Downloading data from a set of tasks (test).
img = test_images[15]

print(img.shape)

# Adding an image
img = (np.expand_dims(img, 0))

print(img.shape)

# Typing the correct image label
predictions_single = probability_model.predict(img)

# Typing the correct label for an image.
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

img = test_images[1]

print(img.shape)

img = (np.expand_dims(img, 0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])

y_pred = model.predict(test_images)
y_p = np.argmax(y_pred, axis=1)
# Confusion matrix
print(confusion_matrix(test_labels, y_p))
