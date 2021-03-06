import os
import time
import logging

import numpy as np
import tensorflow as tf

import model
# import all kinds of neural network here
# import models.cnn
# from models import *
import models

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
# environmental parameters
tf.app.flags.DEFINE_string(
    'outputs_dir', 'outputs/',
    'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer(
    "log_level", 20,
    "numeric value of logging level, 20 for info, 10 for debug.")
tf.app.flags.DEFINE_integer("in_top_k", 1, "compare the top n results.")

# Training parameters
tf.app.flags.DEFINE_integer("num_epochs", 200,
                            "Number of training epochs (default: 100)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_integer(
    "evaluate_every", 100,
    "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100,
                            "Save model after this many steps (default: 100)")

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            "Allow device soft device placement")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of total batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('print_step', 1,
                            """Number of steps to print current state.""")
tf.app.flags.DEFINE_integer('summary_step', 3,
                            """Number of steps to write summaries.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 50,
                            """Number of steps to write checkpoint. """)
tf.app.flags.DEFINE_integer(
    'num_checkpoints', 10,
    "Number of maximum checkpoints to keep. default: 10")
# tf.app.flags.DEFINE_string(
#     "categories", "Arts,Business,Computers,Health",
#     'categories name list, divided by comma, no space in between.')
tf.app.flags.DEFINE_boolean('if_eval', False,
                            "Whether to log device placement.")
tf.app.flags.DEFINE_string('model_type', 'cnn',
                           'the type of NN model. cnn, crnn, resnn, resrnn...')

#########################################
# global variables
#########################################
CUR_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
TRAIN_FOLDER = FLAGS.model_type + '_' + CUR_TIME
if FLAGS.data_dir == "data/":
    TRAIN_DIR = os.path.join(FLAGS.outputs_dir, TRAIN_FOLDER)
else:
    TRAIN_DIR = os.path.join(FLAGS.data_dir, FLAGS.outputs_dir, TRAIN_FOLDER)
SUMMARY_DIR = os.path.join(TRAIN_DIR, "summaries")
CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')

#########################################
# functions
#########################################


def train(model_type):
    """Train neural network for a number of steps."""
    logging.info("\nstart training...")
    with tf.Graph().as_default():
        model_train = model_type(is_train=True)
        if FLAGS.if_eval:
            model_eval = model_type(is_train=False)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(),
                               max_to_keep=FLAGS.num_checkpoints)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=
                                                FLAGS.log_device_placement))
        # with tf.Session(config=tf.ConfigProto(
        #     log_device_placement=FLAGS.log_device_placement)) as sess:
        sess.run(init)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary writer
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        try:
            step = 1
            while not coord.should_stop():
                start_time = time.time()
                loss_value, top_k = model_train.train_step(sess)

                duration = time.time() - start_time

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                # print current state
                if step % FLAGS.print_step == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    precision = np.sum(top_k) / FLAGS.batch_size
                    format_str = (
                        'step %d, loss = %.2f, precision = %.2f (%.1f '
                        'examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value, precision,
                                        examples_per_sec, sec_per_batch))

                # save summary
                if step % FLAGS.summary_step == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    logging.info("step: {}, wrote summaries.".format(step))

                # Save the model checkpoint periodically and eval on test set.
                if step % FLAGS.checkpoint_step == 0 or (
                        step + 1) == FLAGS.max_steps:
                    saver_path = saver.save(sess,
                                            CHECKPOINT_PATH,
                                            global_step=step)
                    logging.info("\nSaved model checkpoint to {}\n".format(
                        saver_path))

                    # start evaluation for this checkpoint, or run eval.py
                    if FLAGS.if_eval:
                        logging.info("\n\nevaluating current checkpoint:")
                        pass

                step += 1
                # sleep for test use
                # print("sleep 1 second...")
                # time.sleep(1)
        except tf.errors.OutOfRangeError:
            print("Done~")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def set_logging(level=logging.INFO,
                stream=False,
                fileh=False,
                filename="example.log"):
    """set basic logging configurations (root logger).
    args:
        stream (bool): whether print log to console.
        fileh (bool): whether write log to file.
        filename (str): the path of log file.
    return:
        configued root logger.
    """
    handlers = []
    level = level
    log_format = '%(asctime)s: %(message)s'

    if stream:
        handlers.append(logging.StreamHandler())
    if fileh:
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(format=log_format, handlers=handlers, level=level)
    return logging.getLogger()


def main(argv=None):
    print("start of main")

    # file handling
    if not os.path.isdir(FLAGS.outputs_dir):
        os.mkdir(FLAGS.outputs_dir)
        print("create outputs folder: {}".format(FLAGS.outputs_dir))
    # os.mkdir(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)

    # loging
    log_file = os.path.join(TRAIN_DIR, "log")
    set_logging(level=FLAGS.log_level,
                stream=True,
                fileh=True,
                filename=log_file)
    logging.info("\nall arguments:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logging.info("{}={}".format(attr.upper(), value))
    logging.info("")

    # file handling
    logging.info("create train outputs folder: {}".format(TRAIN_DIR))
    os.mkdir(CHECKPOINT_DIR)
    logging.info("create checkpoints folder: {}".format(CHECKPOINT_DIR))

    # get model name
    model_type = model.Model
    if FLAGS.model_type == "cnn":
        model_type = models.cnn.CNN
    elif FLAGS.model_type == "crnn":
        model_type = models.crnn.CRNN
    elif FLAGS.model_type == "resnn":
        model_type = models.resnn.ResNN
    elif FLAGS.model_type == "resrnn":
        model_type = models.resrnn.ResRNN
    elif FLAGS.model_type == "rnn":
        model_type = models.rnn.RNN
    elif FLAGS.model_type == "rrnn":
        model_type = models.rnn.RRNN
    else:
        raise ValueError("wrong model_name:" + FLAGS.mode)

    # core
    train(model_type)
    logging.info("summary dir: " + SUMMARY_DIR)

    print("\n end of main")


if __name__ == '__main__':
    tf.app.run()
