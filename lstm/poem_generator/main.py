import os
import numpy as np
import tensorflow as tf
import corpus
import model
import train

tf.flags.DEFINE_integer("batch_size", default=64, help="training batch size")
tf.flags.DEFINE_integer("time_steps", default=48, help="length of each time step")
tf.flags.DEFINE_integer("hidden_size", default=256, help="hidden layer size")
tf.flags.DEFINE_integer("num_epoch", default=100, help="number of training epoches")
tf.flags.DEFINE_float("learning_rate", default=0.05, help="learning rate")
tf.flags.DEFINE_string("run_mode", default="write", help="running mode(train/write)")
tf.flags.DEFINE_string("write_mode", default="acrostic", help="writing mode(default/head/acrostic)")
tf.flags.DEFINE_string("corpus_file",
                       default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "poem.txt"),
                       help="Poem corpus file")
tf.flags.DEFINE_string("model_dir",
                       default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                            "model/"),
                       help="Directory to save model")

FLAGS = tf.flags.FLAGS

def generator(input_, batch_size, encoder):
    while True:
        ind = np.random.randint(0, len(input_), batch_size)
        encoded = np.asarray(
            [[encoder[c] for c in input_[i]] for i in ind],
            dtype=np.int32)
        yield encoded

def compose():
    print("loading corpus from {}".format(FLAGS.corpus_file))
    poems = corpus.process(FLAGS.corpus_file)
    encoder, decoder = corpus.build_vocab(poems)
    vocab_size = len(encoder)
    # generator = corpus.generator(poems, FLAGS.batch_size, encoder)

    if FLAGS.run_mode == "train":
        train.run_training(FLAGS.batch_size, FLAGS.hidden_size, FLAGS.time_steps,
                           FLAGS.learning_rate, FLAGS.num_epoch, vocab_size,
                           poems, generator, encoder, decoder, FLAGS.model_dir, "default")

    if FLAGS.run_mode == "write":
        start_token = encoder['s']
        head = input("Please input head: ")
        head = [encoder[c] for c in head]
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            poem_write = model.write_model(FLAGS.hidden_size, vocab_size, FLAGS.time_steps,
                                           FLAGS.write_mode, head, start_token)

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                sample = sess.run(poem_write)
                sample = ''.join([decoder[c] for c in sample])
                # print("Sample poem:")
                print()
                print("Joeyhaohao, please write a poem for my girl.")
                for i in range(4):
                    print(sample[i*12: (i+1)*12])

if __name__ == "__main__":
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    compose()