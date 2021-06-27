#Librerias
import tensorflow as tf
import tensorflow 
from tensorflow import keras
import os, os.path
from os.path import isfile, join
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# Se carga la base de datos CIFAR-10
from keras.datasets import cifar10
(trainX, trainy), (testX, testy) = cifar10.load_data()

# Datos generales de la base de datos
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# Muestra de las primeras 9 imágenes
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(trainX[i])
plt.show()

from tensorflow.keras.utils import  to_categorical
from tensorflow import constant
# Se usa la codificación one-hot para
trainy = to_categorical(trainy)
testy = to_categorical(testy)

# convierte los valores de pixeles entre el rango [0-1]
def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

from keras.models import Sequential
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
    
    
    # prepare pixel data
trainX, testX = prep_pixels(trainX, testX)
# define model
model = define_model()
# fit model
history = model.fit(trainX, trainy, epochs=20, batch_size=64, validation_data=(testX, testy), verbose=0)
# evaluate model
_, acc = model.evaluate(testX, testy, verbose=0)
print('> %.3f' % (acc * 100.0))
# learning curves
summarize_diagnostics(history)