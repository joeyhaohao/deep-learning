import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference

BATCH_SIZE = 128
LEARNING_RATE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
DATA_PATH = "/tmp/mnist_data/"
MODEL_PATH = '/tmp/fully_connected_nn/'
LOG_PATH = '/tmp/log/'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    with tf.name_scope("input"):
        feature = tf.placeholder(tf.float32, [None, inference.INPUT_SIZE], name="feature")
        label = tf.placeholder(tf.float32, [None, inference.OUTPUT_SIZE], name="label")
    # Visualize in tensorboard
    with tf.name_scope("input_reshape"):
        image_reshape = tf.reshape(feature, [-1, 28, 28, 1])
        tf.summary.image('input', image_reshape, 10)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(feature, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # Use moving average model
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(label, 1), logits=y)
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("losses"))
        tf.summary.scalar("loss", loss)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("train"):
    # Use exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                                   mnist.train.num_examples / BATCH_SIZE,
                                                   LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.group([train_step, variable_averages_op])

    summaries_all = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        train_writer = tf.summary.FileWriter(logdir=LOG_PATH + '/train')
        valid_writer = tf.summary.FileWriter(logdir=LOG_PATH + '/valid')
        test_writer = tf.summary.FileWriter(logdir=LOG_PATH + '/test')
        for i in range(TRAINING_STEPS):
            data_x, data_y = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, summary = sess.run([train_op, loss, global_step, summaries_all],
                                                    feed_dict={feature: data_x, label: data_y})
            train_writer.add_summary(summary, i)
            if i % 1000 == 0:
                print("loss at step {}: {:.3f}".format(i, loss_value))
                valid_acc, summary = sess.run([accuracy, summaries_all],
                                               feed_dict={feature: mnist.validation.images,
                                               label: mnist.validation.labels})
                valid_writer.add_summary(summary, i)
                test_acc, summary = sess.run([accuracy, summaries_all],
                                             feed_dict={feature: mnist.test.images,
                                             label: mnist.test.labels})
                test_writer.add_summary(summary, i)
                print("Valid acc at step {}: {:.3f}".format(i, valid_acc))
                print("Test acc at step {}: {:.3f}".format(i, test_acc))
                # Save training models every 1000 step
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)

    train_writer.close()
    valid_writer.close()
    test_writer.close()

def main(_):
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()
