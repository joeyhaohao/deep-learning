import os
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hanziconv import HanziConv

tfd = tfp.distributions
t2s = HanziConv.toSimplified


class Config(object):
    data_path = "/tmp/poem.txt"
    model_path = "/tmp/model"
    batch_size = 64
    timesteps = 48
    num_write = 4
    hidden_size = 512
    learning_rate = 0.05
    num_epoch = 100

def get_config():
    return Config

def process_poem(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if len(content)==48:
                poems.append("s" + content)
    return poems

def build_vocab(poems):
    words = [word for poem in poems for word in poem]
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    encoder = dict(zip(words, range(len(words))))
    decoder = {v:k for k,v in encoder.items()}
    return encoder, decoder

def generator(input_, batch_size, encoder):
    while True:
        ind = np.random.randint(0, len(input_), batch_size)
        encoded = np.asarray(
            [[encoder[c] for c in input_[i]] for i in ind],
            dtype=np.int32)
        yield encoded

def lstm_cell(x, h, c, name=None, reuse=False):
    nin = x.shape[-1].value
    nout = h.shape[-1].value
    with tf.variable_scope(name, default_name="lstm",
                           values=[x, h, c], reuse=reuse):
        wx = tf.get_variable("kernel/input", [nin, nout * 4],
                             dtype=tf.float32,
                             initializer=tf.orthogonal_initializer(1.0))
        wh = tf.get_variable("kernel/hidden", [nout, nout * 4],
                             dtype=tf.float32,
                             initializer=tf.orthogonal_initializer(1.0))
        b = tf.get_variable("bias", [nout * 4],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(z, 4, axis=1)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f + 1.0)
    o = tf.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    return h, c

def build_model(input_, vocab_size, config):
    x = tf.one_hot(input_, depth=vocab_size, dtype=tf.float32)
    h = tf.zeros([config.batch_size, config.hidden_size])
    c = tf.zeros([config.batch_size, config.hidden_size])
    hs = []
    reuse = None

    for t in range(config.timesteps):
        if t > 0:
            reuse = True
        xt = x[:, t, :]
        h, c = lstm_cell(xt, h, c, name="lstm", reuse=reuse)
        hs.append(h)

    h = tf.stack(hs, axis=1)
    logits = tf.layers.dense(h, vocab_size, name="dense")
    output = tfd.Categorical(logits=logits)
    return output

def write_model(batch_size, vocab_size, config):
    x = tf.random_uniform([batch_size], 0, vocab_size, dtype=tf.int32)
    h = tf.zeros([batch_size, config.hidden_size])
    c = tf.zeros([batch_size, config.hidden_size])
    xs = []
    for _ in range(config.timesteps):
        x = tf.one_hot(x, depth=vocab_size, dtype=tf.float32)
        h, c = lstm_cell(x, h, c, name="lstm")
        logits = tf.layers.dense(h, vocab_size, name="dense")
        x = tfd.Categorical(logits=logits).sample()
        xs.append(x)

    xs = tf.cast(tf.stack(xs, 1), tf.int32)
    return xs


if __name__ == '__main__':
    config = get_config()
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    poems = process_poem(config.data_path)
    encoder, decoder = build_vocab(poems)
    vocab_size = len(encoder)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        input_poem = tf.placeholder(tf.int32, [None, config.timesteps + 1])
        x = input_poem[:, :-1]
        target = input_poem[:, 1:]
        y = build_model(x, vocab_size, config)
        poem_write = write_model(config.num_write, vocab_size, config)

    loss = -tf.reduce_sum(y.log_prob(target))
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    train_op = optimizer.minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(init_op)

        n_iter_per_epoch = len(poems) // config.batch_size
        try:
            for epoch in range(config.num_epoch):
                total_loss = 0.0
                for step in range(n_iter_per_epoch):
                    data = next(generator(poems, config.batch_size, encoder))
                    [_, loss_value] = sess.run([train_op, loss],
                                               feed_dict={input_poem: data})
                    total_loss += loss_value

                    if (step + 1) % 100 == 0:
                        print("Epoch: {}, batch: {}, average training loss: {:0.5f}".format(
                            epoch, step, total_loss / (step * config.batch_size)))

                        samples = sess.run(poem_write)
                        samples = [''.join([decoder[c] for c in sample]) for sample in samples]
                        print("Sample poems:")
                        for sample in samples:
                            print(t2s(sample))
                if epoch % 10 == 0:
                    saver.save(sess, config.model_path, global_step=epoch)
        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint...')
            saver.save(sess, config.model_path, global_step=epoch)
