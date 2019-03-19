import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import train
import inference

# Evaluate accuracy of new model every 10 sec
EVAL_INTERVAL = 10
DATA_PATH = "/tmp/mnist_data/"

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        feature = tf.placeholder(tf.float32, [None, inference.INPUT_SIZE], name="feature")
        label = tf.placeholder(tf.float32, [None, inference.OUTPUT_SIZE], name="label")
        y = inference.inference(feature, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        # variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    valid_acc = sess.run(accuracy,
                                        feed_dict={feature: mnist.validation.images,
                                                   label: mnist.validation.labels})
                    test_acc = sess.run(accuracy,
                                        feed_dict={feature: mnist.test.images,
                                                   label: mnist.test.labels})
                    print("Valid acc at step {}: {}".format(global_step, valid_acc))
                    print("Test acc at step {}: {}".format(global_step, test_acc))
                else:
                    print("No checkpoint found.")
                    return
                time.sleep(EVAL_INTERVAL)

def main(_):
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()