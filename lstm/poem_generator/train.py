import os
import tensorflow as tf
import model

def run_training(batch_size, hidden_size, time_steps, learning_rate, num_epoch, vocab_size,
                 poems, generator, encoder, decoder, model_dir, write_mode):

    start_token = encoder['s']
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        input_poem = tf.placeholder(tf.int32, [None, time_steps + 1])
        x = input_poem[:, :-1]
        target = input_poem[:, 1:]
        y = model.build_model(x, batch_size, hidden_size, vocab_size, time_steps)
        poem_write = model.write_model(hidden_size, vocab_size, time_steps, write_mode,
                                       None, start_token)

    loss = -tf.reduce_sum(y.log_prob(target))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(init_op)

        # checkpoint = tf.train.latest_checkpoint(model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        n_iter_per_epoch = len(poems) // batch_size
        try:
            for epoch in range(num_epoch):
                for step in range(n_iter_per_epoch):
                    data = next(generator(poems, batch_size, encoder))
                    [_, loss_value] = sess.run([train_op, loss],
                                               feed_dict={input_poem: data})

                    if (step + 1) % 10 == 0:
                        print("Epoch: {}, batch: {}, average training loss: {:0.5f}".format(
                            epoch, step, loss_value / batch_size))

                        sample = sess.run(poem_write)
                        sample = ''.join([decoder[c] for c in sample])
                        print("Sample poem:")
                        for i in range(4):
                            print(sample[i*12: (i+1)*12])
                saver.save(sess, os.path.join(model_dir, "model.ckpt"), global_step=epoch)
        except KeyboardInterrupt:
            print('Interrupt manually, try saving checkpoint...')
            saver.save(sess, os.path.join(model_dir, "model.ckpt"), global_step=epoch)
            print('Checkpoint saved.')
