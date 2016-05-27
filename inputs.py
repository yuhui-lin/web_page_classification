"""This code process TFRecords text classification datasets.
YOU MUST run convertbefore running this (but you only need to
run it once).
"""
import os
import time
import logging

import tensorflow as tf

#########################################
# FLAGS
#########################################
# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_integer(
    "embed_length", 50,
    "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_string("tfr_folder", "tfr-time/",
                           'path of TFRecords files directory under data_dir.')
tf.app.flags.DEFINE_integer(
    "node_length", "100",
    'the max #tokens of node string, 100 for 512 characters string.')
tf.app.flags.DEFINE_string(
    "categories", "Arts,Business,Computers,Health",
    'categories name list, divided by comma, no space in between.')

#########################################
# global variables
#########################################
CATEGORIES = FLAGS.categories.split(',')
# Constants used for dealing with the files, matches convert_to_records.
TFR_SUFFIX = '.TFR'


#########################################
# functions
#########################################

def char_index_batch_to_2d_tensor(batch, batch_size, num_labels):
    sparse_labels = tf.reshape(batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    concatenated = tf.concat(1, [indices, sparse_labels])
    concat = tf.concat(0, [[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1, 0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature(
                [], tf.int64),
            'target': tf.VarLenFeature(tf.string),
            'unlabeled': tf.VarLenFeature(tf.string),
            'labeled': tf.VarLenFeature(tf.string),
        })

    t_dense = tf.sparse_tensor_to_dense(features['target'])
    t_decode = tf.decode_raw(t_dense, tf.unit8)
    t_cast = tf.cast(t_decode, tf.float32)
    t_reshape = tf.reshape(t_cast, [-1, FLAGS.embed_length])

    u_dense = tf.sparse_tensor_to_dense(features['unlabeled'])
    u_decode = tf.decode_raw(u_dense, tf.unit8)
    u_cast = tf.cast(u_decode, tf.float32)
    u_reshape = tf.reshape(u_cast, [-1, FLAGS.node_length, FLAGS.embed_length])

    l_dense = tf.sparse_tensor_to_dense(features['labeled'])
    l_decode = tf.decode_raw(l_dense, tf.unit8)
    l_cast = tf.cast(l_decode, tf.int32)
    l_reshape = tf.reshape(l_cast, [-1, len(CATEGORIES)])

    label = tf.cast(features['label'], tf.int32)

    return [t_reshape, u_reshape, l_reshape], label


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
        page, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # shuffling of examples between files, use shuffle_batch_join
        capacity = min_shuffle + 3 * batch_size
        pages, sparse_labels = tf.train.shuffle_batch_join(
            [page, label],
            batch_size=batch_size,
            # num_threads=2,
            capacity=capacity,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_shuffle)
        return pages, sparse_labels
