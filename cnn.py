''' model 1 '''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def get_model(image_shape):
    ''' CNN by NVidia '''
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=image_shape))

    # Conv2D layer
    model.add(Conv2D(3, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.382))

    # Conv2D layer
    model.add(Conv2D(24, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.382))

    # Conv2D layer
    model.add(Conv2D(36, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.382))

    # Conv2D layer
    model.add(Conv2D(48, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.382))

    # Conv2D layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    # Full connected layer
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.382))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.382))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.382))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.382))
    model.add(Dense(1))
    return model
