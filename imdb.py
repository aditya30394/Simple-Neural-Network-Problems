# -*- coding: utf-8 -*-
"""IMDB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/0B1sSVpQymiERZnItX0Y4b1V1SktnOUc4VkVGRUt3M0JoQk04
"""

#import the dataset

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# Encoding the integer sequences into a binary matrix
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

# Define the structure of the model 
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# use sigmoid for the last layer as we need a binary output.

# Compile the model, configure the optimizer
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Split the training set for hyperparameter tuning. 
# Here we try to tune the ideal number of epochs to get the best result
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Training Phase. Record the accuracy and error/loss for tuning later on
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# get the loss and accuracy from history
history_dict = history.history
history_dict.keys()

# Plotting the training and validation loss for tuning the number of epochs
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plotting the accuracy curves
plt.clf()
acc_values = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# The ideal number of epochs looks to be 3. So retrain the model using all the
# training examples (all of 25k)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=512)
results = model.evaluate(x_test, y_test)

# check the final accuracy and loss
results

# make prediction on unseen data now - test data
predictions = model.predict(x_test)
predictions = predictions >= 0.5
# change the datatype to float from bool
predictions = np.asarray(predictions).astype('float32')
# convert it to array
predictions = np.reshape(predictions, -1)
# size of test
print(len(y_test))
# size of prediction - should be equal to test 
print(len(predictions))
correct_predictions = np.sum(y_test == predictions)
print(correct_predictions)
print(float(correct_predictions) / len(y_test))

model.evaluate(x_test, y_test)

