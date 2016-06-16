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
tf.app.flags.DEFINE_string(
    'data_dir', 'data/',
    'Directory to download data files and write the converted result')
tf.app.flags.DEFINE_string(
    'tfr_folder', 'TFR_test',
    'tfr folder name under data_dir')

FLAGS = tf.app.flags.FLAGS
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

print("Done")
