import os
import time
import logging
import collections

import numpy as np
import tensorflow as tf

# import all kinds of neural network here
# import models.cnn
# from models import *
import models

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
# environmental parameters
tf.app.flags.DEFINE_string('model_type', 'cnn',
                           'the type of NN model. cnn, crnn, resnn, resrnn...')
tf.app.flags.DEFINE_string(
    'outputs_dir', 'outputs/',
    'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer(
    "log_level", 20,
    "numeric value of logging level, 20 for info, 10 for debug.")
tf.app.flags.DEFINE_boolean('if_eval', True,
                            "Whether to log device placement.")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 128, "mini Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("in_top_k", 1, "compare the top n results.")
tf.app.flags.DEFINE_integer("num_epochs", 200,
                            "Number of training epochs (default: 100)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of total batches to run.""")

# learning rate decay
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 20,
                            "number of epochs for every learning rate decay.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.33,
                          "learning rate decay factor.")
tf.app.flags.DEFINE_float("initial_lr", 0.1, "inital learning rate.")
tf.app.flags.DEFINE_integer('min_lr', 8, "e^-8, minimum learning rate")

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            "Allow device soft device placement")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('print_step', 1,
                            """Number of steps to print current state.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to write summaries.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 100,
                            """Number of steps to write checkpoint. """)
tf.app.flags.DEFINE_integer(
    'num_checkpoints', 5,
    "Number of maximum checkpoints to keep. default: 10")
tf.app.flags.DEFINE_integer(
    'sleep', 0, "the number of seconds to sleep between steps. 0, 1, 2...")

#########################################
# global variables
#########################################
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value))
CUR_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
TRAIN_FOLDER = FLAGS.model_type + '_' + CUR_TIME
if FLAGS.data_dir == "data/":
    TRAIN_DIR = os.path.join(FLAGS.outputs_dir, TRAIN_FOLDER)
else:
    TRAIN_DIR = os.path.join(FLAGS.data_dir, FLAGS.outputs_dir, TRAIN_FOLDER)
# SUMMARY_DIR = os.path.join(TRAIN_DIR, "summaries")
CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')

# dataset default settings
# see settings in main()
DataConf = collections.namedtuple(
    'DataConf', ['num_train', 'num_test', 'num_cats', 'tfr_folder'])
DATA_CONF = {
    'dmoz-5-2500': DataConf(2000, 500, 5, 'TFR_5-2500'),
    'dmoz-5': DataConf(40000, 10000, 5, 'TFR_5'),
    'dmoz-10': DataConf(40000, 10000, 10, 'TFR_10'),
    'ukwa-10': DataConf(4000, 1000, 10, 'TFR_ukwa'),
}

#########################################
# functions
#########################################


def train(model_type):
    """Train neural network for a number of steps."""
    logging.info("\nstart training...")
    with tf.Graph().as_default():
        # build computing graph
        with tf.variable_scope("model", reuse=None):
            model_train = model_type(is_train=True)
        if FLAGS.if_eval:
            with tf.variable_scope("model", reuse=True):
                model_eval = model_type(is_train=False)

        saver = tf.train.Saver(tf.all_variables(),
                               max_to_keep=FLAGS.num_checkpoints)
        # sv = tf.train.Supervisor(logdir=TRAIN_DIR, saver=saver,
        #                          save_summaries_secs=10, save_model_secs=20)
        sv = tf.train.Supervisor(logdir=TRAIN_DIR,
                                 saver=saver,
                                 save_summaries_secs=0,
                                 save_model_secs=0)

        logging.info("\n")
        logging.info("start building Graph??? (This might take a while)")

        # Start running operations on the Graph.
        # sess = sv.prepare_or_wait_for_session()
        sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement))
        # with sv.managed_session("") as sess:

        logging.info("\n")
        logging.info("start training...")

        try:
            while not sv.should_stop():
                start_time = time.time()
                step, loss_value, top_k = model_train.train_step(sess)
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
                    logging.info(format_str %
                                 (step, loss_value, precision,
                                  examples_per_sec, sec_per_batch))

                # save summary
                if step % FLAGS.summary_step == 0:
                    summary_str = sess.run(sv.summary_op)
                    sv.summary_writer.add_summary(summary_str, step)
                    logging.info("step: {}, wrote summaries.".format(step))

                # Save the model checkpoint periodically and eval on test set.
                if step % FLAGS.checkpoint_step == 0 or (
                        step + 1) == FLAGS.max_steps:
                    saver_path = sv.saver.save(sess,
                                               CHECKPOINT_PATH,
                                               global_step=step)
                    logging.info("\nSaved model checkpoint to {}\n".format(
                        saver_path))

                    # start evaluation for this checkpoint, or run eval.py
                    if FLAGS.if_eval:
                        logging.info("\n\nevaluating current checkpoint:")
                        precision = model_eval.eval_once(sess)
                        print('%s: precision @ 1 = %.3f' %
                              (time.strftime("%c"), precision))

                        summary = tf.Summary()
                        summary.ParseFromString(sess.run(sv.summary_op))
                        summary.value.add(tag='precision @ 1',
                                          simple_value=precision)
                        sv.summary_writer.add_summary(summary, step)
                        print("write eval summary\n\n")

                # sleep for test use
                if FLAGS.sleep > 0:
                    print("sleep {} second...".format(FLAGS.sleep))
                    time.sleep(FLAGS.sleep)
        except tf.errors.OutOfRangeError:
            print("sv checkpoint saved path: " + sv.save_path)
            print("Done~")
        finally:
            sv.request_stop()
        # coord.join(threads)
        sv.wait_for_stop()
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

    # dataset settings !!!!!!!!!
    # shouldn't in train.py,
    # three options: 1. model or input, don't use FLAGS
    # 2. dataset class like inception, or namedtuple
    # 3. pass argument
    data_conf = DATA_CONF[FLAGS.dataset]
    FLAGS.num_train_examples = FLAGS.num_train_examples or data_conf.num_train
    FLAGS.num_test_examples = FLAGS.num_test_examples or data_conf.num_test
    FLAGS.num_cats = FLAGS.num_cats or data_conf.num_cats
    FLAGS.tfr_folder = FLAGS.tfr_folder or data_conf.tfr_folder

    # loging
    log_file = os.path.join(TRAIN_DIR, "log.txt")
    set_logging(level=FLAGS.log_level,
                stream=True,
                fileh=True,
                filename=log_file)
    logging.info("\nall arguments:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logging.info("{}={}".format(attr, value))
    logging.info("")

    # file handling
    logging.info("create train outputs folder: {}".format(TRAIN_DIR))
    os.mkdir(CHECKPOINT_DIR)
    logging.info("create checkpoints folder: {}".format(CHECKPOINT_DIR))

    # get model name
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
        model_type = models.rrnn.RRNN
    # elif FLAGS.model_type == "presnn":
    #     model_type = models.pre_resnn.PreResNN
    else:
        raise ValueError("wrong model_name:" + FLAGS.model_type)

    # core
    train(model_type)
    logging.info("summary dir: " + TRAIN_DIR)

    print("\n end of main")


if __name__ == '__main__':
    tf.app.run()
