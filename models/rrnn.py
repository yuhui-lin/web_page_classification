"""CNN model class"""
import tensorflow as tf
# import model
import models.rnn

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class RRNN(models.rnn.RNN):
    """recurrent neural network model.
    classify web page only based on target html."""


    def rnn(self, sequences):
        """Build the RNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """

        # # [batch_size, html_len, we_dim] to
        # # list[batch_size, we_dim], length:html_len
        # inputs_rnn = [tf.squeeze(input_, [1])
        #               for input_ in tf.split(1, FLAGS.html_len, sequences)]

        # high-level classifier - RNN
        with tf.variable_scope("low_rnn"):
            cell = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.num_cats)
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=sequences,
                                               dtype=tf.float32)

            # _activation_summary(outputs_fw)

        # net_shape = state_fw.get_shape().as_list()
        # softmax_len = net_shape[1]
        # # net = tf.reshape(state_fw, [-1, softmax_len])
        # net = state_fw

        # softmax, i.e. softmax(WX + b)
        # with tf.variable_scope('softmax_linear'):
        #     WW = tf.get_variable(
        #         "WW",
        #         shape=[softmax_len, self.num_cats],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = self._variable_on_cpu('b',
        #                               [self.num_cats],
        #                               tf.constant_initializer(value=0.1))
        #     softmax_linear = tf.nn.xw_plus_b(net, WW, b, name="scores")
        #
        return state


    def inference(self, page_batch):
        """Build the RRNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        self.num_layers = 1
        self.hidden_layers = FLAGS.we_dim

        return self.high_classifier(page_batch, self.rnn)
