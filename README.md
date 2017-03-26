# German Traffic Sign Recognition in Keras

Robert Crowe

v1.0

4 Jan 2017

This is an implementation in Keras on Tensorflow of a convolutional network classifier for the German Traffic Sign
dataset.  You'll need a copy of the dataset, which is available at:

>http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
    
The architecture is a fairly straghtforward CNN, using pooling and dropout.  For a more detailed
model in Tensorflow, see:

>https://github.com/robertcrowe/TrafficSigns

Keras is a nice layer on top of Tensorflow, especially for prototyping or roughing out an approach.  The performance
penalty is minimal, and the abstraction is just right for controlling important things like early stopping while
leaving the code concise and clear.  I'm a fan.
