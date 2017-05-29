import os
from glob import glob
import numpy as np
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from skimage import color, io
from scipy.misc import imresize
from sklearn.model_selection import train_test_split


def train():

    file_path = 'images/'

    malignant_file_path = os.path.join(file_path, 'SOB_M_*.png')
    benign_file_path = os.path.join(file_path, 'SOB_B_*.png')

    malignant_files = sorted(glob(malignant_file_path))
    benign_files = sorted(glob(benign_file_path))

    n_files = len(malignant_files) + len(benign_files)
    print(n_files)

    image_height = 460
    image_width = 700
    colour_channels = 3

    allX = np.zeros((n_files, image_height, image_width, colour_channels), dtype='float64')
    allY = np.zeros(n_files)
    count = 0

    for file in malignant_files:
        try:
            img = io.imread(file)
            new_img = imresize(img, (image_height, image_width, colour_channels))
            allX[count] = np.array(new_img)
            count += 1
        except:
            continue

    for file in benign_files:
        try:
            img = io.imread(file)
            new_img = imresize(img, (image_height, image_width, colour_channels))
            allX[count] = np.array(new_img)
            count += 1
        except:
            continue

    X, X_test, Y, Y_test = train_test_split(allX, allY, test_size=0.1, random_state=42)

    Y = to_categorical(Y, 2)
    Y_test = to_categorical(Y_test, 2)

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()

    network = input_data(shape=[None, image_height, image_width, colour_channels],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    conv_1 = conv_2d(network, 32, colour_channels, activation='relu', name='conv_1')

    network = max_pool_2d(conv_1, 2)

    conv_2 = conv_2d(network, 64, colour_channels, activation='relu', name='conv_2')

    conv_3 = conv_2d(conv_2, 64, colour_channels, activation='relu', name='conv_3')

    network = max_pool_2d(conv_3, 2)

    network = fully_connected(network, 512, activation='relu')

    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    acc = Accuracy(name='Accuracy')

    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate='0.0005', metric=acc)

    model = tflearn.DNN(network, checkpoint_path='cnn_classifier/checkpoints/model.tflearn', max_checkpoints=3,
                        tensorboard_verbose=3, tensorboard_dir='cnn_classifier/tflearn_logs/')

    model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500, n_epoch=100, run_id='model', show_metric=True)

    model.save('model_final.tflearn')