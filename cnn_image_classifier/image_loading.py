import os
import cv2
import glob
import numpy as np
import logging


def load_training(image_dir):

    images = []
    labels = []
    ids = []
    cls = []

    logging.info("Reading training images")

    training_dirs = os.listdir(os.path.join(image_dir, 'train'))

    for category in training_dirs:
        index = training_dirs.index(category)

        logging.debug("Loading %s images (Index: %s)" % (category, index))

        path = os.path.join(image_dir, 'train', category, '*g')
        file_list = glob.glob(path)

        for file in file_list:
            image = cv2.imread(file)
            image = cv2.resize(image, (32, 32), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(training_dirs))
            label[index] = 1.0
            labels.append(label)
            filebase = os.path.basename(file)
            ids.append(filebase)
            cls.append(category)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def read_training_sets(image_dir, image_size, categories, validation_size=0):
    class DataSets:
        pass

    data_sets = DataSets()
