""" eval_result

This is used to train and test the model in main.py, using the test dataset to
assess the accuracy of the model. 
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

import main

def eval_result():
    model = main.main()

    with open('test.p', 'rb') as f:
        testdata = pickle.load(f)
    
    # Preprocess data & one-hot encode the labels
    X_test = testdata['features']
    X_test = X_test.astype('float32')
    X_test = X_test / 255 - 0.5

    n_classes = 43
    y_test = testdata['labels']
    y_test = np_utils.to_categorical(y_test, n_classes)

    # Evaluate model on test data
    result = model.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
    mets = model.metrics_names
    print('\n\n{} = {}, {} = {}'.format(mets[0], result[0], mets[1], result[1]))

if __name__ == "__main__":
    eval_result()