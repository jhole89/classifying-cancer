import tensorflow as tf


def batch_images(file_list, x_pixels, y_pixels, channels):

    filename_queue = tf.train.string_input_producer(file_list['filename'])

    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_png(content, channels=channels)
    image = tf.cast(image, tf.float32)

    resized_image = tf.image.resize_images(image, [y_pixels, x_pixels])

    image_batch, labels = tf.train.batch([resized_image, file_list['label']], batch_size=100)

    return image_batch, labels


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def conv2d(images, weights):
    return tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(images):
    return tf.nn.max_pool(images, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def new_conv_layer(layer, num_input_channels, num_filters, filter_size=3, pooling=True):

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)

    bias = new_bias(length=num_filters)

    layer = conv2d(layer, weights) + bias

    if pooling:
        layer = max_pool_2x2(layer)

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def train(file_list):

    num_classes = 2
    x_pixels = 700
    y_pixels = 460
    colour_channels = 3

    image_pixels = x_pixels * y_pixels * colour_channels

    image_batch, labels = batch_images(file_list, x_pixels, y_pixels, colour_channels)

    with tf.name_scope('input'):
        image_placeholder = tf.placeholder(tf.float32, shape=[None, image_pixels], name='image-input')
        label_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes], name='label-input')

    with tf.name_scope('weights'):
        weight = tf.Variable(tf.zeros([image_pixels, num_classes]))

    with tf.name_scope('biases'):
        bias = tf.Variable(tf.zeros([num_classes]))

    reshaped_images = tf.reshape(image_placeholder, [-1, x_pixels, y_pixels, colour_channels])

    layer1_output_channels = 32

    layer_conv1, weights_conv1 = new_conv_layer(layer=image_placeholder,
                                                num_input_channels=colour_channels,
                                                num_filters=layer1_output_channels)

    layer_conv2, weights_conv2 = new_conv_layer(layer=layer_conv1,
                                                num_input_channels=layer1_output_channels,
                                                num_filters=layer1_output_channels)

    layer_conv3, weights_conv3 = new_conv_layer(layer=layer_conv2,
                                                num_input_channels=layer1_output_channels,
                                                num_filters= 2 * layer1_output_channels)

    layer_flat, num_features = flatten_layer(layer_conv3)

