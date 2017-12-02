''' train '''
import argparse
import csv
from random import shuffle

from os import listdir

import cv2
import numpy as np

import matplotlib.pyplot as plt

from model2 import get_model

IMAGE_SHAPE = (99, 320, 3)
CORRECTION = 0.5
MULTIPLIER = 6   # for each line in csv we can feed 6 images to NN
BATCH_SIZE = 120  # has to be a multple of 6
VALIDATION_RATIO = 0.3

DATA_PATH = '.\\data\\'

TRACK_NAME = 't1'  # or 't2'

DATA_PATH_LIST = [(DATA_PATH + d)
                  for d in listdir(DATA_PATH) if d.startswith(TRACK_NAME)]


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


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--resume', action='store_true')
    ARGS = PARSER.parse_args()

    print("Building model...", flush=True)

    MODEL = get_model(IMAGE_SHAPE)

    MODEL.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    print("Training...", flush=True)
    RECORDS = get_records(DATA_PATH_LIST)

    if ARGS.resume:
        print("Resuming previous training...")
        MODEL.load_weights('model.h5')

    HISTORY = MODEL.fit_generator(
        generator=generator(RECORDS, BATCH_SIZE),
        steps_per_epoch=int((len(RECORDS) * MULTIPLIER *
                             (1 - VALIDATION_RATIO) / BATCH_SIZE)),
        validation_data=generator(RECORDS, BATCH_SIZE, validation=True),
        validation_steps=int(
            (len(RECORDS) * MULTIPLIER * VALIDATION_RATIO) / BATCH_SIZE),
        epochs=20)

    # print the keys contained in the history object
    print(HISTORY.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(HISTORY.history['loss'])
    plt.plot(HISTORY.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    print("Saving model...", flush=True)
    MODEL.save('model.h5')
