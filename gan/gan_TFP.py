"""Trains a GAN to generate synthetic images of MNIST digits.

A Generative Adversarial Network (GAN) is a generative model that learns the
probability distribution of training examples and generates similar samples. It
simultaneously trains two models: a generative model G that learns the data
distribution, and a discriminative model D that estimates the probability that a
sample came from the training data [1].

To learn the generator's distribution p_g over data z, we define a multilayer
perceptron G(z; theta_g), where the prior of input noise z~p_z(z) is a Gaussian
distribution. We also define a second multilayer perceptron D(x; theta_d)
that outputs a single scalar. D(x) represents the probability that x came
from the training data rather than p_g.

We train D to maximize the probability of assigning the correct label to both
training examples and samples from G. We simultaneously train G to maximize the
probability of D making a mistake. This framework corresponds to a minimax
two-player game with value function V(G, D):

```none
V(G, D) = E_{x~p_data(x)}[log(D(x))] + E_{z~p_z(z)}[log(1-D(G(z)))]
```

This optimization problem is bilevel: it requires a minima solution with respect
to generative parameters theta_g and a maxima solution with respect to
discriminative parameters theta_d. In practice, the algorithm proceeds by
iterating gradient updates on each network. The goal of training is to reach
the equilibrium where the generator produces samples that are indistinguishable
by the discriminator.

#### References

[1]: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu,
     David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.
     Generative Adversarial Nets. In _Neural Information Processing
     Systems Conference_, 2014.
     https://arxiv.org/pdf/1406.2661.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
from matplotlib.backends import backend_agg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib.learn.python.learn.datasets import mnist

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float('learning_rate',
                      default=1e-4,
                      help='Initial learning rate.')
flags.DEFINE_integer('max_steps',
                        default=100000,
                        help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                        default=128,
                        help='Batch size.')
flags.DEFINE_integer('hidden_size',
                        default=128,
                        help='Hidden layer size.')
flags.DEFINE_integer('viz_steps',
                     default=1000,
                     help='Frequency at which save generated images.')
flags.DEFINE_string('data_dir',
                    default='/tmp/data',
                    help='Directory where data is stored (if using real data)')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'generative_adversarial_network/'),
    help="Directory to put the model's fit.")
flags.DEFINE_bool('fake_data',
                  default=None,
                  help='If true, uses fake data. Defaults to real data.')

FLAGS = flags.FLAGS


def build_input_pipeline(train_images, batch_size):
  """Build an iterator over training batches."""

  training_dataset = tf.data.Dataset.from_tensor_slices(train_images)
  training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
  images = training_iterator.get_next()
  return images


def build_fake_data(size):
  """Generate fake images of MNIST digits."""

  # Generate random noise from a Gaussian distribution.
  fake_images = np.random.normal(size=size)
  return fake_images


def plot_generated_images(images, fname):
  """Save a synthetic image as a PNG file.

  Args:
    images: samples of synthetic images generated by the generative network.
    fname: Python `str`, filename to save the plot to.
  """
  fig = plt.figure(figsize=(4, 4))
  canvas = backend_agg.FigureCanvasAgg(fig)

  for i, image in enumerate(images):
    ax = fig.add_subplot(4, 4, i + 1)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(image.reshape(IMAGE_SHAPE[:-1]), cmap='Greys_r')

  fig.tight_layout()
  plt.subplots_adjust(wspace=0.05, hspace=0.05)
  canvas.print_figure(fname, format='png')


def main(argv):
  del argv  # unused
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warning(
        'Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  # Collapse the image data dimension for use with a fully-connected layer.
  image_size = np.prod(IMAGE_SHAPE, dtype=np.int32)
  if FLAGS.fake_data:
    train_images = build_fake_data([10, image_size])
  else:
    mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=image_size)
    train_images = mnist_data.train.images

  images = build_input_pipeline(train_images, FLAGS.batch_size)

  # Build a Generative network. We use the Flipout Monte Carlo estimator
  # for the fully-connected layers: this enables lower variance stochastic
  # gradients than naive reparameterization.
  with tf.compat.v1.name_scope('Generator'):
    random_noise = tf.placeholder(tf.float64, shape=[None, FLAGS.hidden_size])
    generative_net = tf.keras.Sequential([
        tfp.layers.DenseFlipout(FLAGS.hidden_size, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(image_size, activation=tf.sigmoid)
    ])
    sythetic_image = generative_net(random_noise)

  # Build a Discriminative network. Define the model as a Bernoulli
  # distribution parameterized by logits from a fully-connected layer.
  with tf.compat.v1.name_scope('Discriminator'):
    discriminative_net = tf.keras.Sequential([
        tfp.layers.DenseFlipout(FLAGS.hidden_size, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(1)
    ])
    logits_real = discriminative_net(images)
    logits_fake = discriminative_net(sythetic_image)
    labels_distribution_real = tfd.Bernoulli(logits=logits_real)
    labels_distribution_fake = tfd.Bernoulli(logits=logits_fake)

  # Compute the model loss for discrimator and generator, averaged over
  # the batch size.
  loss_real = -tf.reduce_mean(
      input_tensor=labels_distribution_real.log_prob(
          tf.ones_like(logits_real)))
  loss_fake = -tf.reduce_mean(
      input_tensor=labels_distribution_fake.log_prob(
          tf.zeros_like(logits_fake)))
  loss_discriminator = loss_real + loss_fake
  loss_generator = -tf.reduce_mean(
      input_tensor=labels_distribution_fake.log_prob(
          tf.ones_like(logits_fake)))

  with tf.compat.v1.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op_discriminator = optimizer.minimize(
        loss_discriminator,
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
    train_op_generator = optimizer.minimize(
        loss_generator,
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

  with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.max_steps + 1):
      # Iterate gradient updates on each network.
      _, loss_value_d = sess.run([train_op_discriminator, loss_discriminator],
                                 feed_dict={random_noise: build_fake_data(
                                     [FLAGS.batch_size, FLAGS.hidden_size])})
      _, loss_value_g = sess.run([train_op_generator, loss_generator],
                                 feed_dict={random_noise: build_fake_data(
                                     [FLAGS.batch_size, FLAGS.hidden_size])})

      # Visualize some sythetic images produced by the generative network.
      if step % FLAGS.viz_steps == 0:
        images = sess.run(sythetic_image,
                          feed_dict={random_noise: build_fake_data(
                              [16, FLAGS.hidden_size])})

        plot_generated_images(images, fname=os.path.join(
            FLAGS.model_dir,
            'step{:06d}_images.png'.format(step)))

        print('Step: {:>3d} Loss_discriminator: {:.3f} '
              'Loss_generator: {:.3f}'.format(step, loss_value_d, loss_value_g))


if __name__ == '__main__':
  tf.compat.v1.app.run()
