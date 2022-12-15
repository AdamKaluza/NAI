"""
==========================================
Teaching a neural network to classify data from CIFAR10 dataset
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install ssl
pip install matplotlib
pip install tensorflow
==========================================
Usage:
python animals.py
==========================================
"""
import ssl

import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Downloading and preparing the CIFAR10 collection

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# We normalize the data by dividing it by 255
train_images, test_images = train_images / 255.0, test_images / 255.0

# data verification, class names

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Displaying sample images from the database
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # CIFAR labels are arrays, so we need an additional index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

'''
We define a model, this model is suitable for a simple layer stack, where each layer
has one input tesor and one output tesor.
'''

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Let's show the model's architecture so far

model.summary()

# Add thick layers on top

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Full model architecture

model.summary()

# Compilation and training of the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Model evaluation

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
