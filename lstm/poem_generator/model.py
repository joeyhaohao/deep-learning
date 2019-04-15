import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

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

def build_model(input_, batch_size, hidden_size, vocab_size, time_steps):
    x = tf.one_hot(input_, depth=vocab_size, dtype=tf.float32)
    h = tf.zeros([batch_size, hidden_size])
    c = tf.zeros([batch_size, hidden_size])
    hs = []
    reuse = None

    for t in range(time_steps):
        if t > 0:
            reuse = True
        xt = x[:, t, :]
        h, c = lstm_cell(xt, h, c, name="lstm", reuse=reuse)
        hs.append(h)

    h = tf.stack(hs, axis=1)
    logits = tf.layers.dense(h, vocab_size, name="dense")
    output = tfd.Categorical(logits=logits)
    return output

def write_model(hidden_size, vocab_size, time_steps, write_mode,
                head, start_token):
    x = tf.constant([start_token])
    h = tf.zeros([1, hidden_size])
    c = tf.zeros([1, hidden_size])
    xs = []

    for i in range(time_steps):
        x = tf.one_hot(x, depth=vocab_size, dtype=tf.float32)
        h, c = lstm_cell(x, h, c, name="lstm")
        if write_mode == "head" and i == 0:
            x = [head]
        elif write_mode == "acrostic" and i % 12 == 0:
            x = [head[i // 12]]
        else:
            logits = tf.layers.dense(h, vocab_size, name="dense")
            x = tfd.Categorical(logits=logits).sample()
        xs.append(tf.reshape(x, []))

    return xs