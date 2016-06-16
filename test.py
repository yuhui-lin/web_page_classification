import os
import time
import logging

import numpy as np
import tensorflow as tf

import inputs

# from cnn import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
# environmental parameters
tf.app.flags.DEFINE_string(
    'outputs_dir', 'cnn/outputs',
    'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer(
    "log_level", 20,
    "numeric value of logging level, 20 for info, 10 for debug.")
tf.app.flags.DEFINE_integer("in_top_k", 1, "compare the top n results.")
# import tensorflow as tf
#
# FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

FLAGS.batch_size
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print(FLAGS.batch_size)

print("Done")
