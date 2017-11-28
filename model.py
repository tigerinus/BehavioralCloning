''' train '''
import csv
import cv2
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

DATA_PATH_LIST = [
    '.\\data\\t1r1n',
    '.\\data\\t1r2r',
    #'.\\data\\t2r1n'
]

CORRECTION = 0.5

IMAGES = []
MEASUREMENTS = []

for data_path in DATA_PATH_LIST:
    csv_path = data_path + '\\driving_log.csv'
    with open(csv_path) as csvfile:
        print('Reading image file listed in \'' +
              csv_path + '\' and the measurements...')

        reader = csv.reader(csvfile)
        for line in reader:
            # image - center
            source_path = line[0]
            filename = source_path.split('\\')[-1]
            current_path = data_path + '\\IMG\\' + filename
            image = cv2.imread(current_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            IMAGES.append(image[crop_offset:, ])
            measurement = float(line[3])
            MEASUREMENTS.append(measurement)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            IMAGES.append(image[crop_offset:, ])
            MEASUREMENTS.append(-measurement)

            # image - left
            source_path = line[1]
            filename = source_path.split('\\')[-1]
            current_path = data_path + '\\IMG\\' + filename
            image = cv2.imread(current_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            IMAGES.append(image[crop_offset:, ])
            MEASUREMENTS.append(measurement + CORRECTION)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            IMAGES.append(image[crop_offset:, ])
            MEASUREMENTS.append(-measurement)

            # image - right
            source_path = line[2]
            filename = source_path.split('\\')[-1]
            current_path = data_path + '\\IMG\\' + filename
            image = cv2.imread(current_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            IMAGES.append(image[crop_offset:, ])
            MEASUREMENTS.append(measurement - CORRECTION)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            IMAGES.append(image[crop_offset:, ])
            MEASUREMENTS.append(-measurement)

assert IMAGES[0].shape == (99, 320, 3)
#for i in range(6):
#    plt.imshow(IMAGES[i])
#    plt.show()


X_TRAIN = np.array(IMAGES)
Y_TRAIN = np.array(MEASUREMENTS)

MODEL = Sequential()
MODEL.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGES[0].shape))

# Conv2D layer
MODEL.add(Conv2D(9, (7, 7)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D())

# Conv2D layer
MODEL.add(Conv2D(27, (5, 5)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D())

# Conv2D layer
MODEL.add(Conv2D(81, (3, 3)))
MODEL.add(Activation('relu'))
MODEL.add(MaxPooling2D())

# Full connected layer
MODEL.add(Flatten())
MODEL.add(Dense(300))
MODEL.add(Activation('relu'))
MODEL.add(Dense(150))
MODEL.add(Activation('relu'))
MODEL.add(Dropout(0.5))
MODEL.add(Dense(1))

MODEL.compile(loss='mse', optimizer='adam')

MODEL.fit(X_TRAIN, Y_TRAIN, validation_split=0.3, shuffle=True, epochs=5)

MODEL.save('model.h5')
