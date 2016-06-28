"""CNN model class"""
from collections import namedtuple

import tensorflow as tf
# from tensorflow.contrib import learn
from tensorflow.contrib.layers import convolution2d
from tensorflow.contrib.layers import batch_norm

import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class ResNN(model.Model):
    """Residual neural network model.
    classify web page only based on target html."""

    def resnn(self, sequences):
        """Build the resnn model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """

        # [batch_size, html_len, 1, we_dim]
        target_expanded = tf.expand_dims(sequences, 2)

        # Configurations for each bottleneck block.
        BottleneckBlock = namedtuple(
            'BottleneckBlock',
            ['num_layers', 'num_filters', 'bottleneck_size'])
        # blocks = [BottleneckBlock(3, 128, 32),
        #           BottleneckBlock(3, 256, 64),
        #           BottleneckBlock(3, 512, 128),
        #           BottleneckBlock(3, 1024, 256)]
        # blocks = [BottleneckBlock(3, 128, 32),
        #           BottleneckBlock(3, 256, 64)]
        blocks = [BottleneckBlock(3, 64, 32), BottleneckBlock(6, 128, 64),
                  BottleneckBlock(3, 256, 128)]
        # BottleneckBlock(3, 512, 256)]

        # First convolution expands to 64 channels
        with tf.variable_scope('conv_layer1'):
            net = convolution2d(target_expanded,
                                64,
                                [7, 1],
                                stride=[2, 1],
                                activation_fn=self.activation,
                                normalizer_fn=batch_norm,
                                normalizer_params={'decay': self.norm_decay})

        # Max pool
        net = tf.nn.max_pool(net,
                             [1, 3, 1, 1],
                             strides=[1, 2, 1, 1],
                             padding='SAME')

        # First chain of resnets
        with tf.variable_scope('conv_layer2'):
            net = convolution2d(net,
                                blocks[0].num_filters,
                                [1, 1],
                                padding='VALID',
                                activation_fn=self.activation,
                                normalizer_fn=batch_norm,
                                normalizer_params={'decay': self.norm_decay})

        # Create each bottleneck building block for each layer
        for block_i, block in enumerate(blocks):
            for layer_i in range(block.num_layers):

                name = 'block_%d/layer_%d' % (block_i, layer_i)

                # 1x1 convolution responsible for reducing dimension
                with tf.variable_scope(name + '/conv_in'):
                    conv = convolution2d(
                        net,
                        block.bottleneck_size,
                        [1, 1],
                        padding='VALID',
                        activation_fn=self.activation,
                        normalizer_fn=batch_norm,
                        normalizer_params={'decay': self.norm_decay})

                with tf.variable_scope(name + '/conv_bottleneck'):
                    conv = convolution2d(
                        conv,
                        block.bottleneck_size,
                        [3, 1],
                        padding='SAME',
                        activation_fn=self.activation,
                        normalizer_fn=batch_norm,
                        normalizer_params={'decay': self.norm_decay})

                # 1x1 convolution responsible for restoring dimension
                with tf.variable_scope(name + '/conv_out'):
                    conv = convolution2d(
                        conv,
                        block.num_filters,
                        [1, 1],
                        padding='VALID',
                        activation_fn=self.activation,
                        normalizer_fn=batch_norm,
                        normalizer_params={'decay': self.norm_decay})

                # shortcut connections that turn the network into its counterpart
                # residual function (identity shortcut)
                net = conv + net

            try:
                # upscale to the next block size
                next_block = blocks[block_i + 1]
                with tf.variable_scope('block_%d/conv_upscale' % block_i):
                    net = convolution2d(
                        net,
                        next_block.num_filters,
                        [1, 1],
                        activation_fn=self.activation,
                        normalizer_fn=batch_norm,
                        normalizer_params={'decay': self.norm_decay})
            except IndexError:
                pass

        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net,
                             ksize=[1, net_shape[1], net_shape[2], 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID')

        net_shape = net.get_shape().as_list()
        softmax_len = net_shape[1] * net_shape[2] * net_shape[3]
        net = tf.reshape(net, [-1, softmax_len])

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
        """Build the resnn model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        self.activation = tf.nn.relu
        self.norm_decay = 0.99
        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        return self.resnn(target_batch)
