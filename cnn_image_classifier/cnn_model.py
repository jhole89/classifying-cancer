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


def new_fc_layer(layer, num_inputs, num_outputs, relu=True):

    weights = new_weights([num_inputs, num_outputs])
    bias = new_bias(num_outputs)

    layer = tf.matmul(layer, weights) + bias

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def train(train_file_list, test_file_list, log_path='tmp/tensorflow/cnn/logs/cnn_with_summaries'):

    sess = tf.InteractiveSession()
    num_classes = 2
    x_pixels = 700
    y_pixels = 460
    colour_channels = 3

    image_pixels = x_pixels * y_pixels * colour_channels

    with tf.name_scope('input'):
        image_placeholder = tf.placeholder(tf.float32, shape=[None, image_pixels], name='image-input')
        label_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes], name='label-input')

    reshaped_images = tf.reshape(image_placeholder, [-1, x_pixels, y_pixels, colour_channels])

    layer1_output_channels = 32

    layer_conv1, weights_conv1 = new_conv_layer(layer=reshaped_images,
                                                num_input_channels=colour_channels,
                                                num_filters=layer1_output_channels)

    layer_conv2, weights_conv2 = new_conv_layer(layer=layer_conv1,
                                                num_input_channels=layer1_output_channels,
                                                num_filters=layer1_output_channels)

    layer_conv3, weights_conv3 = new_conv_layer(layer=layer_conv2,
                                                num_input_channels=layer1_output_channels,
                                                num_filters= 2 * layer1_output_channels)

    layer_flat, num_features = flatten_layer(layer_conv3)

    layer_fc1 = new_fc_layer(layer_flat, num_features, num_outputs=128)

    layer_fc2 = new_fc_layer(layer_fc1, num_inputs=128, num_outputs=num_classes, relu=False)

    with tf.name_scope('softmax'):
        model_prediction = tf.nn.softmax(layer_fc2)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=label_placeholder))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(model_prediction, 1), tf.argmax(label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('cost', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    train_images, train_labels = batch_images(train_file_list, x_pixels, y_pixels, colour_channels)
    test_images, test_labels = batch_images(test_file_list, x_pixels, y_pixels, colour_channels)

    for i in range(20000):

        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={image_placeholder: train_images, label_placeholder: train_labels})

            print("step %d, training accuracy %g" % (i, train_accuracy))

        optimizer.run(feed_dict={image_placeholder: train_images, label_placeholder: train_labels})

    # evaluate model
    print("test accuracy %g" % accuracy.eval(
        feed_dict={image_placeholder: test_images, label_placeholder: test_labels}))
