import numpy as np
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


def train():

    training_path = 'images/train'

    image_height = 64
    image_width = 64
    colour_channels = 3

    X, Y = image_preloader(
        training_path,
        image_shape=(image_height, image_width),
        mode='folder',
        categorical_labels=True,
        normalize=True)

    X = np.reshape(X, (-1, image_height, image_width, colour_channels))

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    network = input_data(shape=[None, image_height, image_width, colour_channels],
                         data_preprocessing=img_prep,
                         name='input')

    network = conv_2d(network, 32, 3, activation='relu', name='conv_1')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu', name='conv_2')
    network = conv_2d(network, 64, 3, activation='relu', name='conv_3')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')

    network = regression(
        network,
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate='0.001')

    model = tflearn.DNN(
        network,
        checkpoint_path='tmp/tflearn/cnn/checkpoints/model.tflearn',
        max_checkpoints=3,
        tensorboard_verbose=3,
        tensorboard_dir='tmp/tflearn/cnn/logs/')

    model.fit(
        X, Y,
        validation_set=0.2,
        n_epoch=1000,
        shuffle=True,
        batch_size=100,
        run_id='model',
        snapshot_step=500)

    model.save('tmp/tflearn/cnn/model/model_final.tflearn')
