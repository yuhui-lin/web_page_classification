"""pre-activation Residual network model class"""
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.layers import convolution2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers.python.layers import utils

import model

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class ResNN(model.Model):
    """Residual neural network model.
    classify web page only based on target html."""

    def BN_ReLU(self, net):
        # Batch Normalization and ReLU
        # 'gamma' is not used as the next layer is ReLU
        net = batch_norm(net,
                         center=True,
                         scale=False,
                         activation_fn=tf.nn.relu, )
        # net = tf.nn.relu(net)
        # activation summary ??
        return net

    def conv1d(self, net, num_ker, ker_size, stride):
        # 1D-convolution
        net = convolution2d(
            net,
            num_outputs=num_ker,
            kernel_size=[ker_size, 1],
            stride=[stride, 1],
            padding='SAME',
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=variance_scaling_initializer(),
            weights_regularizer=l2_regularizer(self.weight_decay),
            biases_initializer=tf.zeros_initializer)
        return net

    def residual_unit(self, net, group_i, unit_i):
        """pre-activation Residual Units from
        https://arxiv.org/abs/1603.05027."""
        name = 'group_%d/unit_%d' % (group_i, unit_i)
        group = self.groups[group_i]

        if group.is_downsample and unit_i == 0:
            stride1 = 2
        else:
            stride1 = 1

        def conv_pre(name, net, num_ker, kernel_size, stride, conv_i):
            """ 1D pre-activation convolution.
            args:
                num_ker (int): number of kernels (out_channels).
                ker_size (int): size of 1D kernel.
                stride (int)
            """
            with tf.variable_scope(name):
                if not (self.special_first and
                        group_i == unit_i == conv_i == 0):
                    net = self.BN_ReLU(net)

                # 1D-convolution
                net = self.conv1d(net, num_ker, kernel_size, stride)
            return net

        def conv_post(name, net, num_ker, kernel_size, stride, conv_i):
            """ 1D post-activation convolution.
            args:
                num_ker (int): number of kernels (out_channels).
                ker_size (int): size of 1D kernel.
                stride (int)
            """
            with tf.variable_scope(name):
                # 1D-convolution
                net = self.conv1d(net, num_ker, kernel_size, stride)
                net = self.BN_ReLU(net)
            return net

        ### residual function
        net_residual = net
        if self.unit_type == 0 and not self.special_first:
            unit_conv = conv_post
        elif self.unit_type == 1:
            unit_conv = conv_pre
        else:
            raise ValueError("wrong residual unit type:{}".format(
                self.unit_type))
        # 1x1 convolution responsible for reducing dimension
        net_residual = unit_conv(name + '/conv_reduce', net_residual,
                                 group.reduced_ker, 1, stride1, 0)
        # 3x1 convolution bottleneck
        net_residual = unit_conv(name + '/conv_bottleneck', net_residual,
                                 group.reduced_ker, 3, 1, 1)
        # 1x1 convolution responsible for restoring dimension
        net_residual = unit_conv(name + '/conv_restore', net_residual,
                                 group.num_ker, 1, 1, 2)

        ### shortcut connection
        num_ker_in = utils.last_dimension(net.get_shape(), min_rank=4)
        if self.shortcut == 0 and unit_i == 0:
            # average pooling for data downsampling
            if group.is_downsample:
                net = tf.nn.avg_pool(net,
                                     ksize=[1, 2, 1, 1],
                                     strides=[1, 2, 1, 1],
                                     padding='SAME')
            # zero-padding for increasing kernel numbers
            if group.num_ker / num_ker_in == 2:
                net = tf.pad(net, [[0, 0], [0, 0], [0, 0],
                                   [int(num_ker_in / 2), int(num_ker_in / 2)]])
            elif group.num_ker != num_ker_in:
                raise ValueError("illigal kernel numbers at group {} unit {}"
                                 .format(group_i, unit_i))
        elif self.shortcut == 1 and unit_i == 0 or self.shortcut == 2:
            # projection
            net = self.conv1d(net, group.num_ker, 1, stride1)

        ### element-wise addition
        net = net + net_residual

        return net

    def resnn(self, sequences):
        """Build the resnn model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """

        # [batch_size, html_len, 1, we_dim]
        target_expanded = tf.expand_dims(sequences, 2)

        # Configurations for each group
        # several residual units (aka. bottleneck blocks) form a group
        UnitsGroup = namedtuple(
            'UnitsGroup',
            [
                'num_units',  # number of residual units
                'num_ker',  # number of kernels for each convolution
                'reduced_ker',  # number of reduced kernels
                'is_downsample'  # (bool): downsample data using stride 2
                # types of BottleneckBlock ??
                # wide resnet kernel*k ??
            ])
        self.groups = [
            UnitsGroup(3, 64, 16, True),
            UnitsGroup(3, 128, 32, True),
            UnitsGroup(3, 128, 64, True),
            # UnitsGroup(6, 128, 64, True),
        ]
        # special first residual unit from P14 of (arxiv.org/abs/1603.05027)
        self.special_first = False
        # shortcut connection type: (arXiv:1512.03385)
        # 0: 0-padding and average pooling
        # 1: convolution projection only for increasing dimension
        # 2: projection for all shortcut
        self.shortcut = 0
        # weight decay
        self.weight_decay = 0.0001
        # the type of residual unit
        # 0: post-activation; 1: pre-activation
        self.unit_type = 1
        # RoR enable level 1
        # requirement: every group is downsampling
        self.ror_l1 = False
        # RoR enable level 2
        self.ror_l2 = False

        # First convolution
        with tf.variable_scope('conv_layer1'):
            net = self.conv1d(target_expanded, self.groups[0].num_ker, 6, 2)
            # if self.special_first:
            net = self.BN_ReLU(net)

        # Max pool
        net = tf.nn.max_pool(net,
                             [1, 3, 1, 1],
                             strides=[1, 2, 1, 1],
                             padding='SAME')

        # stacking Residual Units
        if self.ror_l1:
            net_l1 = net
        for group_i, group in enumerate(self.groups):
            for unit_i in range(group.num_units):
                if self.ror_l2:
                    net_l2 = net
                net = self.residual_unit(net, group_i, unit_i)
                if self.ror_l2:
                    net_l2 = self.conv1d(net_l2, self.groups[group_i].num_ker,
                                         1, 2)
                    net = net + net_l2
        if self.ror_l1:
            net_l1 = self.conv1d(net_l1, self.groups[-1].num_ker, 1,
                                 len(self.groups))
            net = net + net_l1

        # an extra activation before average pooling
        if self.special_first:
            net = self.BN_ReLU(net)

        # padding should be VALID for global average pooling
        # output: batch*1*1*channels
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net,
                             ksize=[1, net_shape[1], net_shape[2], 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID')

        net_shape = net.get_shape().as_list()
        softmax_len = net_shape[1] * net_shape[2] * net_shape[3]
        net = tf.reshape(net, [-1, softmax_len])

        # 1D-fully connected nueral network
        with tf.variable_scope('FC-layer'):
            net = fully_connected(
                net,
                num_outputs=self.num_cats,
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=variance_scaling_initializer(),
                weights_regularizer=l2_regularizer(self.weight_decay),
                biases_initializer=tf.zeros_initializer, )

        return net

    def inference(self, page_batch):
        """Build the resnn model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        # self.activation = tf.nn.relu
        # self.norm_decay = 0.99
        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        return self.resnn(target_batch)
