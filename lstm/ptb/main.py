import os
import sys
import numpy as np
import tensorflow as tf

reader_path = "../"
sys.path.append(os.path.abspath(reader_path))
import reader

class Config(object):
    # The PTB dataset comes from Tomas Mikolov's webpage:
    # http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    data_path = "/tmp/ptb_data"
    init_scale = 0.05
    hidden_size = 256
    num_layers = 2
    vocab_size = 10000
    learning_rate = 0.05
    batch_size = 16
    num_steps = 32
#     eval_batch_size = 1
#     eval_num_steps = 1
    num_epoch = 2
    keep_prob = 0.5
    max_grad_norm = 5


class PTBModel(object):
    def __init__(self, is_train, config, data):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, self.batch_size, self.num_steps)
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size

        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)  # batch_size*num_steps*hidden_size
        if is_train and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_lstm_graph(inputs, config, is_train)

        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        self.cost = tf.reduce_sum(loss)
        self.final_state = state

        if not is_train:
            return

        # 控制梯度膨胀
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, train_vars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, train_vars),
        )

    def _build_lstm_graph(self, inputs, config, is_train):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,
                                                 state_is_tuple=True)
        # Use deep RNN with dropout
        if is_train and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        self.initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self.initial_state
        outputs = []
        with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                #                 if time_step > 0: tf.get_variable_scope().reuse_variables()
                output, state = cell(inputs[:, time_step, :], state)
                outputs.append(output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state


def get_config():
    return Config

def run_epoch(sess, model, train_op=None):
    total_costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    for step in range(model.epoch_size):
        fetches = {
            "cost": model.cost,
            "final_state": model.final_state
        }
        if train_op is not None:
            fetches["train_op"] = train_op
        res = sess.run(fetches, feed_dict={model.initial_state: state})
        cost = res["cost"]
        state = res["final_state"]

        total_costs += cost
        iters += model.num_steps

        if step % 200 == 0:
            print("Perplexity at step %d is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)

def main(_):
    config = get_config()
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(config.data_path)
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("train"):
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = PTBModel(is_train=True, config=config, data=train_data)
        tf.summary.scalar("Training Loss", train_model.cost)
    #         tf.summary.scalar("Learning Rate", model.learning_rate)

    with tf.name_scope("validation"):
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = PTBModel(is_train=False, config=config, data=valid_data)
        tf.summary.scalar("Validation Loss", valid_model.cost)

    with tf.name_scope("test"):
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = PTBModel(is_train=False, config=config, data=test_data)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        #         sess.run(tf.global_variables_initializer())
        #         threads = tf.train.start_queue_runners()
        for i in range(config.num_epoch):
            train_perplexity = run_epoch(sess, train_model, train_model.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(sess, valid_model)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(sess, test_model)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__=="__main__":
    tf.app.run()