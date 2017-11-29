''' train '''
import csv
from random import shuffle

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

IMAGE_SHAPE = (99, 320, 3)
CORRECTION = 0.5
MULTIPLIER = 6   # for each line in csv we can feed 6 images to NN
BATCH_SIZE = 120  # has to be a multple of 6
VALIDATION_RATIO = 0.3

DATA_PATH_LIST = [
    '.\\data\\t1r1n',
    '.\\data\\t1r2r',
    '.\\data\\t1r3n',
    '.\\data\\t2r1n',
    '.\\data\\t2r2r'
]

def get_records(path_list):
    ''' aggregate data from csv files '''
    result = []
    for data_path in path_list:
        csv_path = data_path + '\\driving_log.csv'
        with open(csv_path) as csvfile:
            print('Aggregating image files listed in \'' +
                  csv_path + '\'...', flush=True)
            result.extend(list(csv.reader(csvfile)))
            csvfile.close()
    return result

def split_records(records, validation_ratio=0.3):
    ''' split the records into training data and validation data '''
    split_point = int(len(records) * (1 - validation_ratio))
    return (records[:split_point], records[split_point + 1:])

def generator(records, batch_size, validation=False):
    ''' generator '''

    if validation is False:
        # Shuffle the records and get those for training
        shuffle(records)
        lines, _ = split_records(records)
    else:
        # Get records for validation. No need to shuffle
        _, lines = split_records(records)

    batched_images = []
    batched_measurements = []

    while True:
        for line in lines:
            # image - center
            source_path = line[0]
            image = cv2.imread(source_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            batched_images.append(image[crop_offset:, ])
            measurement = float(line[3])
            batched_measurements.append(measurement)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            batched_images.append(image[crop_offset:, ])
            batched_measurements.append(-measurement)

            # image - left
            source_path = line[1]
            image = cv2.imread(source_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            batched_images.append(image[crop_offset:, ])
            batched_measurements.append(measurement + CORRECTION)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            batched_images.append(image[crop_offset:, ])
            batched_measurements.append(-measurement)

            # image - right
            source_path = line[2]
            image = cv2.imread(source_path)
            crop_offset = int(image.shape[0] * (1 - 0.618))
            batched_images.append(image[crop_offset:, ])
            batched_measurements.append(measurement - CORRECTION)

            # augmenttation - flipped image
            image = cv2.flip(image, 1)
            batched_images.append(image[crop_offset:, ])
            batched_measurements.append(-measurement)

            if len(batched_images) == batch_size:
                yield (np.array(batched_images), np.array(batched_measurements))
                batched_images = []
                batched_measurements = []


print("Building model...", flush=True)

MODEL = Sequential()
MODEL.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGE_SHAPE))

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

print("Training...", flush=True)
RECORDS = get_records(DATA_PATH_LIST)

MODEL.fit_generator(
    generator=generator(RECORDS, BATCH_SIZE),
    steps_per_epoch=int((len(RECORDS) * MULTIPLIER * (1 - VALIDATION_RATIO) / BATCH_SIZE)),
    validation_data=generator(RECORDS, BATCH_SIZE, validation=True),
    validation_steps=int((len(RECORDS) * MULTIPLIER * VALIDATION_RATIO) / BATCH_SIZE),
    epochs=3)

print("Saving model...", flush=True)
MODEL.save('model.h5')
