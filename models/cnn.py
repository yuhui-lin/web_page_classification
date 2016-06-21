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

    def inference(self, page_batch):
        """Build the CNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        target_batch, unlabeled_batch, labeled_batch = page_batch
        filter_sizes = [3, 4, 5]
        num_filters = len(filter_sizes)
        sequence_length = FLAGS.html_len
        # [batch_size, html_len, 1, we_dim]
        embedded_chars_expanded = tf.expand_dims(target_batch, 2)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 1, FLAGS.we_dim, num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape,
                                        stddev=0.1),
                    name="W")
                b = tf.Variable(
                    tf.constant(0.1,
                                shape=[num_filters]),
                    name="b")
                conv = tf.nn.conv2d(embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear'):
            WW = tf.get_variable(
                "WW",
                shape=[num_filters_total, self.num_cats],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_cats]), name="b")
            softmax_linear = tf.nn.xw_plus_b(h_drop, WW, b, name="scores")

        return softmax_linear
