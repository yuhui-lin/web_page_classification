"""This code process TFRecords text classification datasets.
YOU MUST run convertbefore running this (but you only need to
run it once).
"""
import os
import logging

import tensorflow as tf

#########################################
# FLAGS
#########################################
# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
# parameters that related to read dataset will be kept here.

# dataset parameters
tf.app.flags.DEFINE_string(
    'data_dir', 'data/',
    'Directory to download data files and write the converted result')
tf.app.flags.DEFINE_string(
    'tfr_folder', 'TFR_test',
    'tfr folder name under data_dir')
tf.app.flags.DEFINE_integer(
    'we_dim', 50,
    'word embedding dimensionality. 50, 100, 150, 200...')
tf.app.flags.DEFINE_integer(
    'num_read_threads', 5,
    'number of reading threads to shuffle examples between files.')
tf.app.flags.DEFINE_integer(
    "html_len", 512,
    "the number of tokens in one html vector.")
tf.app.flags.DEFINE_integer(
    "num_cats", 10,
    "the nuber of categories of dataset.")
tf.app.flags.DEFINE_integer(
    "num_train_examples", 200,
    "number of training examples per epoch.")
tf.app.flags.DEFINE_integer(
    "num_test_examples", 50,
    "number of testing examples per epoch.")

#########################################
# global variables
#########################################
# CATEGORIES = FLAGS.categories.split(',')
# Constants used for dealing with the files, matches convert_to_records.
TFR_SUFFIX = '.TFR'


#########################################
# functions
#########################################

def read_and_decode(filename_queue):
    """read data from one file and decode to tensors."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature(
                [], tf.int64),
            'target': tf.FixedLenFeature([], tf.string),
            'unlabeled': tf.VarLenFeature(tf.string),
            'labeled': tf.VarLenFeature(tf.string),
        })

    t_dense = features['target']
    # decode it using the same numpy type in convert !!
    t_decode = tf.decode_raw(t_dense, tf.float32)
    # set_shape and reshape are both necessary!!!
    t_decode.set_shape([FLAGS.html_len *  FLAGS.we_dim])
    # t_cast = tf.cast(t_decode, tf.float32)
    t_reshape = tf.reshape(t_decode, [FLAGS.html_len, FLAGS.we_dim])

    # u_dense = tf.sparse_tensor_to_dense(features['unlabeled'])
    # u_decode = tf.decode_raw(u_dense, tf.unit8)
    # u_cast = tf.cast(u_decode, tf.float32)
    # u_reshape = tf.reshape(u_cast, [-1, FLAGS.node_length, FLAGS.embed_length])
    #
    # l_dense = tf.sparse_tensor_to_dense(features['labeled'])
    # l_decode = tf.decode_raw(l_dense, tf.unit8)
    # l_cast = tf.cast(l_decode, tf.int32)
    # l_reshape = tf.reshape(l_cast, [-1, FLAGS.num_cats])

    label = tf.cast(features['label'], tf.int32)


    u_reshape = tf.zeros([3, 4], tf.int32)
    l_reshape = tf.zeros([3, 4], tf.int32)
    return t_reshape, u_reshape, l_reshape, label


def inputs(datatype, batch_size, num_epochs=None, min_shuffle=1):
    """Reads input data num_epochs times.
    Args:
      datatype: Selects between the 'train' and 'test' data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    tfr_dir = os.path.join(FLAGS.data_dir, FLAGS.tfr_folder, datatype)
    if not os.path.isdir(tfr_dir):
        raise FileNotFoundError(tfr_dir)
    tfr_files = [os.path.join(tfr_dir, f) for f in os.listdir(tfr_dir)]
    if not tfr_files:
        raise FileNotFoundError("no tfr file exists under {}".format(tfr_dir))
    logging.info("Reading examples from folder: {}\n".format(tfr_dir))

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(tfr_files,
                                                        num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        page_list = [read_and_decode(filename_queue)] * FLAGS.num_read_threads
        # page, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # shuffling of examples between files, use shuffle_batch_join
        capacity = min_shuffle + 3 * batch_size
        t_batch, u_batch, l_batch, label_batch = tf.train.shuffle_batch_join(
            page_list,
            batch_size=batch_size,
            capacity=capacity,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_shuffle)
        # return t_batch, u_batch, l_batch, label_batch
        return [t_batch, u_batch, l_batch], label_batch
