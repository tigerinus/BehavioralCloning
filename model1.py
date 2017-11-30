''' model 1 '''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def get_model(image_shape):
    ''' definition '''
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=image_shape))

    # Conv2D layer
    model.add(Conv2D(9, (7, 7)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(27, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(81, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    # Full connected layer
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
