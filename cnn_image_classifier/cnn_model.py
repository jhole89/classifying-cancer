import os
import logging
import tensorflow as tf
from cnn_image_classifier.image_loading import read_training_sets


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))


def conv2d(layer, weights):
    return tf.nn.conv2d(input=layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(layer):
    return tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def new_conv_layer(layer, num_input_channels, filter_size, num_filters, use_pooling=True):

    weights = weight_variable(shape=[filter_size, filter_size, num_input_channels, num_filters])

    biases = bias_variable(shape=[num_filters])

    layer = conv2d(layer, weights) + biases

    if use_pooling:
        layer = max_pool_2x2(layer)

    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

    return layer, num_features


def new_fully_connected_layer(layer, num_inputs, num_outputs, use_relu=True):

    weights = weight_variable(shape=[num_inputs, num_outputs])

    biases = bias_variable(shape=[num_outputs])

    layer = tf.matmul(layer, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def print_progress(session, accuracy, epoch, test_feed_dict, val_loss):

    acc = session.run(accuracy, feed_dict=test_feed_dict)

    msg = "Epoch {0} --- Accuracy: {1:>6.1%}, Validation Loss: {2:.3f}"
    logging.info(msg.format(epoch+1, acc, val_loss))


def train(data_dir, model_dir):

    img_size = 32
    neurons = 2 * img_size
    colour_channels = 3
    num_classes = 2
    filter_size = 3
    batch_size = 128
    training_epochs = 500
    log_dir = model_dir + '/tensorflow/cnn/logs/cnn_with_summaries'
    checkpoint_dir = model_dir + '/tensorflow/cnn/model'

    data = read_training_sets(data_dir, img_size, validation_size=.2)

    flat_img_size = img_size * img_size * colour_channels

    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32, shape=[None, flat_img_size], name='x-input')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    with tf.name_scope('reshaping'):
        x_image = tf.reshape(x, [-1, img_size, img_size, colour_channels])
        image_summary = tf.summary.image('example_images', x_image)

    with tf.name_scope('Conv1'):
        layer_conv1 = new_conv_layer(
            x_image,
            num_input_channels=colour_channels,
            filter_size=filter_size,
            num_filters=img_size
        )

    with tf.name_scope('Conv2'):
        layer_conv2 = new_conv_layer(
            layer_conv1,
            num_input_channels=img_size,
            filter_size=filter_size,
            num_filters=neurons
        )

    with tf.name_scope('Conv3'):
        layer_conv3 = new_conv_layer(
            layer_conv2,
            num_input_channels=neurons,
            filter_size=filter_size,
            num_filters=neurons
        )

    with tf.name_scope('Fully_Connected1'):

        flat_layer, num_features = flatten_layer(layer_conv3)

        layer_fc1 = new_fully_connected_layer(
            flat_layer,
            num_features,
            num_outputs=1024
        )

    with tf.name_scope('Fully_Connected2'):

        layer_fc2 = new_fully_connected_layer(
            layer_fc1,
            num_inputs=1024,
            num_outputs=num_classes,
            use_relu=False
        )

    with tf.name_scope('softmax'):
        y_pred = tf.nn.softmax(layer_fc2)

    with tf.name_scope('cross_entropy'):
        y_true_cls = tf.argmax(y_true, dimension=1)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=layer_fc2)
        cost = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        training_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if os.path.exists(checkpoint_dir):
        logging.info("Loading resource: CNN Model [%s]", os.path.join(os.getcwd(), checkpoint_dir + '/model.ckpt'))
        saver.restore(sess, checkpoint_dir + '/model.ckpt')
    else:
        os.makedirs(checkpoint_dir)
        tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    for epoch in range(training_epochs):

        batch_count = int(data.train.num_examples / batch_size)

        for i in range(batch_count):

            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_test_batch, y_test_batch, _, cls_test_batch = data.test.next_batch(batch_size)

            x_batch = x_batch.reshape(batch_size, flat_img_size)
            x_test_batch = x_test_batch.reshape(batch_size, flat_img_size)

            feed_dict_test = {x: x_test_batch,
                              y_true: y_test_batch}

            _, summary = sess.run([training_op, summary_op],
                                  feed_dict={x: x_batch, y_true: y_true_batch})

            writer.add_summary(summary, epoch * batch_count + i)

        if epoch % 50 == 0:
            val_loss = sess.run(cost, feed_dict=feed_dict_test)
            print_progress(sess, accuracy, epoch, feed_dict_test, val_loss)
            save_path = saver.save(sess, checkpoint_dir + '/model.ckpt')
            logging.info("Creating resource: CNN Model [%s]", os.path.join(os.getcwd(), save_path))
