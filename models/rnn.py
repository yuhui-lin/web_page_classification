"""RNN model class"""
import tensorflow as tf
import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class RNN(model.Model):
    """recurrent neural network model.
    classify web page only based on target html."""

    def inference(self, page_batch):
        """Build the RNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        target_batch, unlabeled_batch, labeled_batch = page_batch

        num_layers = 1
        hidden_layers = FLAGS.html_len
        num_local = 1024

        # list[html_len, batch_size, we_dim]
        inputs_rnn = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, FLAGS.we_dim, target_batch)]

        with tf.variable_scope("BiRNN_FW"):
            cell_fw = tf.nn.rnn_cell.GRUCell(hidden_layers)
            cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers)
            initial_state_fw = cells_fw.zero_state(FLAGS.batch_size,
                                                   tf.float32)
            outputs_fw, state_fw = tf.nn.rnn(cells_fw,
                                             inputs_rnn,
                                             initial_state=initial_state_fw)
            # _activation_summary(outputs_fw)

        with tf.variable_scope("BiRNN_BW") as scope:
            cell_bw = tf.nn.rnn_cell.GRUCell(hidden_layers)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers)
            initial_state_bw = cells_bw.zero_state(FLAGS.batch_size,
                                                   tf.float32)
            outputs_tmp, state_bw = tf.nn.rnn(cells_bw,
                                              inputs_rnn[::-1],
                                              initial_state=initial_state_bw)
            # [100, 128, 50]
            # output_bw = _reverse_seq(outputs_tmp, FLAGS.embed_length)
            outputs_bw = outputs_tmp[::-1]
            # _activation_summary(outputs_bw)

        with tf.variable_scope('concat') as scope:

            # [100, 128, 150]
            # change list to tensor
            outputs = tf.concat(
                2, [tf.pack(tensor)
                    for tensor in [outputs_fw, inputs_rnn, outputs_bw]])
            # -> [128, 100, 150] -> [-1, 150]
            dim = FLAGS.we_dim + 2 * hidden_layers
            xi = tf.reshape(tf.transpose(outputs, perm=[1, 0, 2]), [-1, dim])

        # local1
        with tf.variable_scope('local1') as scope:
            weights = self._variable_with_weight_decay('weights',
                                                       shape=[dim, 1024],
                                                       stddev=0.02,
                                                       wd=None)
            biases = self._variable_on_cpu('biases', [1024],
                                           tf.constant_initializer(0.02))
            local1 = tf.nn.tanh(
                tf.matmul(xi, weights) + biases,
                name=scope.name)
            self._activation_summary(local1)

        local1_reshape = tf.reshape(local1, [FLAGS.batch_size, FLAGS.we_dim,
                                             num_local])
        local1_expand = tf.expand_dims(local1_reshape, -1)
        # pool1
        pool1 = tf.nn.max_pool(local1_expand,
                               ksize=[1, FLAGS.we_dim, 1, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID',
                               name='pool1')

        # [128, 1024]

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            pool1_squeeze = tf.squeeze(pool1)
            weights = self._variable_with_weight_decay(
                'weights',
                [num_local, FLAGS.num_cats],
                stddev=0.02,
                wd=None)
            biases = self._variable_on_cpu('biases', [FLAGS.num_cats],
                                           tf.constant_initializer(0.02))
            softmax_linear = tf.add(
                tf.matmul(pool1_squeeze, weights),
                biases,
                name=scope.name)
            self._activation_summary(softmax_linear)
        return softmax_linear
