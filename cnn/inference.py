import tensorflow as tf

INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

def inference(input_tensor, train=False, regularizer=None):
    with tf.variable_scope("conv1"):
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [32],
                                       initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [64],
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.reshape(pool2, [tf.shape(pool2)[0], -1])

    with tf.variable_scope("dense1"):
        dense1_weights = tf.get_variable("weight", [7*7*64, 512],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        dense1_biases = tf.get_variable("biases", [512],
                                        initializer=tf.constant_initializer(0.0))
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(dense1_weights))
        dense1 = tf.nn.relu(tf.matmul(pool2, dense1_weights) + dense1_biases)
        # Use dropout only in fully connected layer
        if train:
            dense1 = tf.nn.dropout(dense1, 0.5)

    with tf.variable_scope("dense2"):
        dense2_weights = tf.get_variable("weight", [512, 10],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        dense2_biases = tf.get_variable("biases", [10],
                                        initializer=tf.constant_initializer(0.0))
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(dense2_weights))
        logit = tf.matmul(dense1, dense2_weights) + dense2_biases
    return logit