import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference

BATCH_SIZE = 128
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
DATA_PATH = "/tmp/mnist_data/"
MODEL_PATH = '/tmp/model/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    feature = tf.placeholder(tf.float32, [BATCH_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.NUM_CHANNELS], name="feature")
    label = tf.placeholder(tf.float32, [None, inference.OUTPUT_SIZE], name="label")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(feature, True, regularizer)

    global_step = tf.Variable(0, trainable=False)
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(label, 1), logits=y)
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_op = tf.group([train_step, variable_averages_op])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            data_x, data_y = mnist.train.next_batch(BATCH_SIZE)
            # Reshape image to [28, 28, 1]
            data_x = np.reshape(data_x, [BATCH_SIZE,
                                         inference.IMAGE_SIZE,
                                         inference.IMAGE_SIZE,
                                         inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_step, loss, global_step],
                                           feed_dict={feature: data_x, label: data_y})

            if i % 1000 == 0:
                print("loss at step {}: {}".format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)

def main(_):
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()