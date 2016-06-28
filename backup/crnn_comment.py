"""CNN model class"""
import tensorflow as tf
import numpy as np
# import model
import models.cnn

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class CRNN(models.cnn.CNN):
    """convolutional neural network model.
    classify web page only based on target html."""

    def high_classifier(self, page_batch, low_classifier):
        """high level classifier."""
        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        # with tf.variable_scope("low_classifier", reuse=None):
        #     # [batch_size, num_cats]
        #     target_logits = low_classifier(target_batch)
        #     # [batch_size, 1, num_cats]
        #     target_exp = tf.expand_dims(target_logits, 1)
        #
        # # unlabeled relatives
        # # list[batch_size], type: int
        # with tf.variable_scope("low_classifier", reuse=True):
        #     # un_len_list = [tf.squeeze(input_, [0])
        #     #             for input_ in tf.split(0, FLAGS.batch_size, un_len)]
        #     # un_batch_list = [tf.squeeze(input_, [0]) for input_ in tf.split(0, FLAGS.batch_size, un_rel)]
        #     # for i in range(len(un_batch_list)):
        #     #     un_list = []
        #     #     un_rel = tf.reshape(un_batch_list[i], tf.pack([un_len_list[i], FLAGS.html_len, FLAGS.we_dim]))
        #     #     # call low_classifier to classify relatives
        #     #     # all relatives of one target composed of one batch
        #     #     un_logits = low_classifier(un_rel)
        #     #     # [batch_size, num_len(variant), num_cats]
        #     #     un_list.append(un_logits)
        #
        #     un_rel = tf.sparse_tensor_to_dense(un_batch)
        #     un_rel = tf.reshape(un_rel, [FLAGS.batch_size, -1, FLAGS.html_len,
        #                                  FLAGS.we_dim])
        #     # call low_classifier to classify relatives
        #     # all relatives of one target composed of one batch
        #     # [batch_size, num_len(variant), num_cats]
        #     un_rel = tf.map_fn(low_classifier, un_rel)

        # input_values = tf.constant([1, 2, 3, 4, 5, 6], name="data")
        # input_values = np.random.randn(FLAGS.batch_size, 1, FLAGS.html_len, FLAGS.we_dim)
        # input_values = tf.convert_to_tensor(input_values, tf.float32)
        # input_values = tf.map_fn(low_classifier, input_values, name="map_fn")

        with tf.variable_scope("low_classifier") as low_scope:
            # [batch_size, 1, html_len, we_dim]
            target_exp = tf.expand_dims(target_batch, 1)
            # [batch_size, 1, num_cats]
            target_logits = tf.map_fn(low_classifier, target_exp, name="map_fn")

            # reuse parameters for low_classifier
            low_scope.reuse_variables()

            un_rel = tf.sparse_tensor_to_dense(un_batch)
            un_rel = tf.reshape(un_rel, [FLAGS.batch_size, -1, FLAGS.html_len,
                                            FLAGS.we_dim])
            # call low_classifier to classify relatives
            # all relatives of one target composed of one batch
            # [batch_size, num_len(variant), num_cats]
            un_rel = tf.map_fn(low_classifier, un_rel, name="map_fn")

        # labeled relatives
        la_rel = tf.sparse_tensor_to_dense(la_batch)
        la_rel = tf.reshape(la_rel, [FLAGS.batch_size, -1, FLAGS.num_cats])

        # concat all inputs for high-level classifier RNN
        # concat_inputs = tf.concat(1, [un_rel, target_logits])
        concat_inputs = tf.concat(1, [un_rel, la_rel, target_logits])

        # number of pages for each target
        num_pages = tf.add(
            tf.add(un_len, la_len),
            tf.ones(
                [FLAGS.batch_size],
                dtype=tf.int32))

        # inputs = np.random.randn(FLAGS.batch_size, 3, FLAGS.num_cats)
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        # high-level classifier - RNN
        with tf.variable_scope("dynamic_rnn"):
            cell = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.num_cats)
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=concat_inputs,
                                               sequence_length=num_pages,
                                               dtype=tf.float32)

        # target_logits = tf.squeeze(target_logits, [1])
        # input_values = tf.squeeze(input_values, [1])
        # return target_logits
        return state

    def inference(self, page_batch):
        """Build the CNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        self.filter_sizes = [3, 4, 5]
        self.num_filters = len(self.filter_sizes)
        self.sequence_length = FLAGS.html_len

        return self.high_classifier(page_batch, self.cnn)
