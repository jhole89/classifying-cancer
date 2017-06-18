import os
import cv2
import glob
import numpy as np
import logging
from sklearn.utils import shuffle
from cnn_image_classifier.DataSet import DataSet


def load_data(image_dir, image_size):

    images = []
    labels = []
    ids = []
    cls = []

    image_dir = os.path.abspath(image_dir)
    binary_cls_map = {}

    logging.info("Loading resource: Images [%s]", image_dir)

    training_dirs = os.listdir(image_dir)

    for category in training_dirs:
        index = training_dirs.index(category)
        binary_cls_map[index] = category

        logging.debug("Loading resource: %s images [Index: %s]" % (category, index))

        path = os.path.join(image_dir, category, '*g')
        file_list = glob.glob(path)

        for file in file_list:
            image = cv2.imread(file)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
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

    return images, labels, ids, cls, binary_cls_map


def read_img_sets(image_dir, image_size, validation_size=0):
    class DataSets:
        pass

    data_sets = DataSets()

    images, labels, ids, cls, cls_map = load_data(image_dir, image_size)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    test_images = images[:validation_size]
    test_labels = labels[:validation_size]
    test_ids = ids[:validation_size]
    test_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.test = DataSet(test_images, test_labels, test_ids, test_cls)

    return data_sets, cls_map
