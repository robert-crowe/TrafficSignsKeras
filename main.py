""" German Traffic Sign Recognition in Keras

Robert Crowe
v1.0
4 Jan 2017

This is an implementation of a convolutional network classifier for the German Traffic Sign
dataset.  You'll need a copy of the dataset, which is available at:

    http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

The architecture is a fairly straghtforward CNN, using pooling and dropout.  For a more detailed
model in Tensorflow, see:

https://github.com/robertcrowe/TrafficSigns

"""

import pickle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.pooling import MaxPooling2D
from keras.datasets import cifar10

def main():
    # split the dataset
    # X_train, X_val, y_train, y_val = train_test_split(
    #     data['features'], data['labels'], test_size=0.33, random_state=42)

    # read the data file
    # with open('train.p', 'rb') as f:
    #     data = pickle.load(f)
    (X_train, y_train), (X_val, y_val) = cifar10.load_data()

    # data normalization
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train = X_train / 255 - 0.5
    X_val = X_val / 255 - 0.5

    # define the model
    batch_size = 32
    n_epochs = 20

    Y_train = np_utils.to_categorical(y_train, 43)
    Y_val = np_utils.to_categorical(y_val, 43)

    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(43, activation='softmax'))

    model.summary() # prints a nice summary

    # Compile and train the model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=n_epochs,
                        verbose=1, validation_data=(X_val, Y_val))
    
    return model

if __name__ == "__main__":
    main()