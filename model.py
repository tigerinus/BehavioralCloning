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
    '.\\data\\t1r1n'
]

IMAGES = []
MEASUREMENTS = []

for data_path in DATA_PATH_LIST:
    csv_path = data_path + '\\driving_log.csv'
    with open(csv_path) as csvfile:
        print('Reading image file listed in \'' +
              csv_path + '\' and the measurements', end='')

        reader = csv.reader(csvfile)
        for line in reader:
            # image
            source_path = line[0]
            filename = source_path.split('\\')[-1]
            current_path = data_path + '\\IMG\\' + filename
            image = cv2.imread(current_path)
            IMAGES.append(image[int(image.shape[0] * (1 - 0.618)):, ])

            # measurement
            measurement = float(line[3])
            MEASUREMENTS.append(measurement)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            IMAGES.append(image[int(image.shape[0] * (1 - 0.618)):, ])

            # augmentation - inversed measurement
            MEASUREMENTS.append(-measurement)

            print('.', end='')

assert IMAGES[0].shape == (99, 320, 3)
# plt.imshow(IMAGES[0])
# plt.show()

# plt.imshow(IMAGES[1])
# plt.show()

X_TRAIN = np.array(IMAGES)
Y_TRAIN = np.array(MEASUREMENTS)

MODEL = Sequential()
MODEL.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGES[0].shape))

# Conv2D layer
MODEL.add(Conv2D(6, (3, 3)))
MODEL.add(MaxPooling2D())
MODEL.add(Dropout(0.5))
MODEL.add(Activation('relu'))

# Conv2D layer
MODEL.add(Conv2D(16, (3, 3)))
MODEL.add(MaxPooling2D())
MODEL.add(Dropout(0.5))
MODEL.add(Activation('relu'))

# Full connected layer
MODEL.add(Flatten())
MODEL.add(Dense(80))
MODEL.add(Dense(20))
MODEL.add(Dense(1))

MODEL.compile(loss='mse', optimizer='adam')

MODEL.fit(X_TRAIN, Y_TRAIN, validation_split=0.3,
          shuffle=True, epochs=10, verbose=1)

MODEL.save('model.h5')
