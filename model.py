''' train '''
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

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
            assert image.shape == (160, 320, 3)
            IMAGES.append(image)
            # measurement
            measurement = float(line[3])
            MEASUREMENTS.append(measurement)
            print('.', end='')

X_TRAIN = np.array(IMAGES)
Y_TRAIN = np.array(MEASUREMENTS)

MODEL = Sequential()
MODEL.add(Flatten(input_shape=IMAGES[0].shape))
MODEL.add(Dense(1))

MODEL.compile(loss='mse', optimizer='adam')

MODEL.fit(X_TRAIN, Y_TRAIN, validation_split=0.2,
          shuffle=True, nb_epoch=30, verbose=1)

MODEL.save('model.h5')
