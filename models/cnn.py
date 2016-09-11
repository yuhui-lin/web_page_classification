"""CNN model class"""
import tensorflow as tf
import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class CNN(model.Model):
    """simple convolutional neural network model.
    classify web page only based on target html."""

    def cnn(self, sequences):
        """
        Args:
            sequences: [batch_size, html_len, we_dim]
        """
        # [batch_size, html_len, 1, we_dim]
        sequ_exp = tf.expand_dims(sequences, 2)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                W = self._variable_with_weight_decay(
                    "W", [filter_size, 1, FLAGS.we_dim, self.num_filters], 0.1)
                b = self._variable_on_cpu('b',
                                          [self.num_filters],
                                          tf.constant_initializer(value=0.1))
                conv = tf.nn.conv2d(sequ_exp,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self._activation_summary(h)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear'):
            WW = tf.get_variable(
                "WW",
                shape=[self.num_filters_total, self.num_cats],
                initializer=tf.contrib.layers.xavier_initializer())
            bb = self._variable_on_cpu('bb',
                                       [self.num_cats],
                                       tf.constant_initializer(value=0.1))
            softmax_linear = tf.nn.xw_plus_b(h_drop, WW, bb, name="scores")

        return softmax_linear

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

        target_batch, un_batch, un_len, la_batch, la_len = page_batch
        return self.cnn(target_batch)
