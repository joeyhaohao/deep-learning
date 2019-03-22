import tensorflow as tf

INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10

def summaries(var, name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name, stddev)

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", shape=[INPUT_SIZE, HIDDEN_SIZE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        summaries(weights, 'layer1/weights')
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(weights))
        biases = tf.get_variable("biases", [HIDDEN_SIZE],
                                 initializer=tf.constant_initializer(0.0))
        summaries(weights, 'layer1/biases')
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        summaries(weights, 'layer1/activation')
        tf.summary.histogram('layer1/activations', layer1)

    with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", shape=[HIDDEN_SIZE, OUTPUT_SIZE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(weights))
        biases = tf.get_variable("biases", [OUTPUT_SIZE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2