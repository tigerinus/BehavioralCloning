''' model 1 '''
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Cropping2D, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def get_model(image_shape):
    ''' CNN by NVidia '''
    model = Sequential()

    model.add(Cropping2D(cropping=(
        (int(image_shape[0] * (1 - 0.618)), 0), (0, 0)), input_shape=image_shape))
    model.add(BatchNormalization(axis=1))

    # Conv2D layer
    model.add(Conv2D(3, (5, 5)))
    model.add(ELU())
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(24, (5, 5)))
    model.add(ELU())
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(36, (5, 5)))
    model.add(ELU())
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(48, (3, 3)))
    model.add(ELU())
    model.add(MaxPooling2D())

    # Conv2D layer
    model.add(Conv2D(64, (3, 3)))
    model.add(ELU())

    # Full connected layer
    model.add(Flatten())
    model.add(Dense(1164, kernel_regularizer=regularizers.l2()))
    model.add(ELU())
    model.add(Dropout(1 - 0.618))
    model.add(Dense(100, kernel_regularizer=regularizers.l2()))
    model.add(ELU())
    model.add(Dropout(1 - 0.618))
    model.add(Dense(50, kernel_regularizer=regularizers.l2()))
    model.add(ELU())
    model.add(Dropout(1 - 0.618))
    model.add(Dense(10, kernel_regularizer=regularizers.l2()))
    model.add(ELU())
    model.add(Dropout(1 - 0.618))
    model.add(Dense(1, kernel_regularizer=regularizers.l2()))
    return model
