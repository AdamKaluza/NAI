"""
==========================================
Teaching a neural network to classify data from ionosphere.data dataset
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install numpy
pip install matplotlib
pip install sklearn
pip install tensorflow
==========================================
Usage:
python neural_network_ionosphere.py
==========================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

# Load the dataset
df = pd.read_csv('ionosphere.data', header=None)

# Split into input and output
X_ionosphere, y_ionosphere = df.values[:, :-1], df.values[:, -1]

# Ensure that all data are floating point values
X_ionosphere = X_ionosphere.astype('float32')

# Strings to int
y_ionosphere = LabelEncoder().fit_transform(y_ionosphere)

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_ionosphere, y_ionosphere, test_size=0.33)

# Determine the number of input
ionosphere_features = X_ionosphere.shape[1]

# Model
model = Sequential()
model.add(Dense(50, activation='relu', kernel_initializer='he_normal', input_shape=(ionosphere_features,)))
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))
pred_test = model.predict(X_test)
pred_test = np.round(pred_test).astype(int)

# Calculate accuracy score
score = accuracy_score(y_test, pred_test)

# Print summary statistics
print('Accuracy: %.3f' % score)

# Plot learning curves
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='value')
plt.legend()
plt.show()
