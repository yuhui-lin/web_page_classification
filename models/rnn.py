"""RNN model class"""
import tensorflow as tf
import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_rnn_layers", 2, "number of rnn layers")


class RNN(model.Model):
    """recurrent neural network model.
    classify web page only based on target html."""

    def test_bi_rnn(self, page_batch):
        """Build the RNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        target_batch, unlabeled_batch, labeled_batch = page_batch

        num_rnn_layers = 1
        hidden_layers = FLAGS.we_dim
        num_local = 1024

        # [batch_size, html_len, 1, we_dim] to
        # list[batch_size, we_dim], length:html_len
        inputs_rnn = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, FLAGS.html_len, target_batch)]

        with tf.variable_scope("BiRNN_FW"):
            cell_fw = tf.nn.rnn_cell.GRUCell(hidden_layers)
            cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_rnn_layers)
            initial_state_fw = cells_fw.zero_state(FLAGS.batch_size,
                                                   tf.float32)
            outputs_fw, state_fw = tf.nn.rnn(cells_fw,
                                             inputs_rnn,
                                             initial_state=initial_state_fw)
            # _activation_summary(outputs_fw)

        with tf.variable_scope("BiRNN_BW") as scope:
            cell_bw = tf.nn.rnn_cell.GRUCell(hidden_layers)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_rnn_layers)
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
                                                       shape=[dim, num_local],
                                                       stddev=0.02,
                                                       wd=None)
            biases = self._variable_on_cpu('biases', [num_local],
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

    def rnn(self, sequences):
        """Build the RNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """

        # [batch_size, html_len, we_dim] to
        # list[batch_size, we_dim], length:html_len
        # inputs_rnn = [tf.squeeze(input_, [1])
        #               for input_ in tf.split(1, FLAGS.html_len, sequences)]

        with tf.variable_scope("RNN"):
            cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_layers)
            cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] *
                                                   FLAGS.num_rnn_layers)
            # initial_state_fw = cells_fw.zero_state(FLAGS.batch_size,
            #                                        tf.float32)
            outputs_fw, state_fw = tf.nn.dynamic_rnn(cells_fw,
                                                     inputs=sequences,
                                                     dtype=tf.float32)
            # _activation_summary(outputs_fw)

        net_shape = state_fw.get_shape().as_list()
        softmax_len = net_shape[1]
        # net = tf.reshape(state_fw, [-1, softmax_len])
        net = state_fw

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear'):
            WW = tf.get_variable(
                "WW",
                shape=[softmax_len, self.num_cats],
                initializer=tf.contrib.layers.xavier_initializer())
            b = self._variable_on_cpu('b',
                                      [self.num_cats],
                                      tf.constant_initializer(value=0.1))
            softmax_linear = tf.nn.xw_plus_b(net, WW, b, name="scores")

        return softmax_linear

    def inference(self, page_batch):
        """Build the RNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        self.hidden_layers = FLAGS.we_dim

        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        return self.rnn(target_batch)
