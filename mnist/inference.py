import tensorflow as tf

INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer!=None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        weights = get_weight_variable([INPUT_SIZE, HIDDEN_SIZE], regularizer)
        biases = tf.get_variable("biases", [HIDDEN_SIZE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        weights = get_weight_variable([HIDDEN_SIZE, OUTPUT_SIZE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_SIZE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2