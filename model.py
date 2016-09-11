"""neural network model."""
import math

import tensorflow as tf
import numpy as np

import inputs

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS

# parameters applied for both train.py and eval.py will be kept here.
# currently don't put any flag here
# only in_top_k for both train and eval

#########################################
# global variables
#########################################
# FEATURE_NUM = 256

# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
# TOWER_NAME = 'tower'


#########################################
# functions
#########################################
class Model(object):
    """super class for neural network models."""

    def __init__(self, is_train=False):
        self.num_cats = FLAGS.num_cats
        self.NUM_EPOCHS_PER_DECAY = FLAGS.num_epochs_per_decay
        self.LEARNING_RATE_DECAY_FACTOR = FLAGS.lr_decay_factor
        self.INITIAL_LEARNING_RATE = FLAGS.initial_lr
        self.min_lr = 0.1**FLAGS.min_lr

        if is_train:
            # build training graph
            self.dropout_keep_prob = FLAGS.dropout_keep_prob

            self.global_step = tf.get_variable(
                "global_step",
                initializer=tf.zeros_initializer(
                    [],
                    dtype=tf.int64),
                trainable=False)

            # get input data
            page_batch, label_batch = self.inputs_train()

            # Build a Graph that computes the logits predictions from the
            self.logits = self.inference(page_batch)

            # Calculate predictions.
            self.top_k_op = tf.nn.in_top_k(self.logits, label_batch, 1)
            tf.scalar_summary("accuracy",
                              tf.reduce_mean(tf.cast(self.top_k_op, "float")))

            # Calculate loss.
            self.loss = self.loss(self.logits, label_batch)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            self.train_op = self.training(self.loss, self.global_step)
        else:
            # build eval graph
            self.dropout_keep_prob = 1

            page_batch_eval, label_batch_eval = self.inputs_eval()
            self.logits_eval = self.inference(page_batch_eval)
            # Calculate predictions.
            self.top_k_op_eval = tf.nn.in_top_k(self.logits_eval,
                                                label_batch_eval, 1)
            tf.scalar_summary(
                "accuracy_eval (batch)",
                tf.reduce_mean(tf.cast(self.top_k_op_eval, "float")))

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        # Error: these summaries cause high classifier error!!!
        # All inputs to node MergeSummary/MergeSummary must be from the same frame.

        # tensor_name = re.sub('%s_[0-9]*/' % "tower", '', x.op.name)
        # tf.histogram_summary(tensor_name + '/activations', x)
        # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd=None):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            # weight_decay = tf.mul(tf.constant(0.1), wd, name='weight_loss')
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            # tf.add_to_collection('losses', wd)
        return var

    def inputs_train(self):
        """Construct input examples for training process.
        Returns:
            sequences: 4D tensor of [batch_size, 1, input_length, alphabet_length] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        return inputs.inputs("train",
                             FLAGS.batch_size,
                             FLAGS.num_epochs,
                             min_shuffle=1000)

    def inputs_eval(self):
        """Construct input examples for evaluations.
        similar to inputs_train
        """
        # don't shuffle
        return inputs.inputs("test", FLAGS.batch_size, None, min_shuffle=1)

    def inference(self, page_batch):
        raise NotImplementedError("Should have implemented this")
        return

    def high_classifier(self, page_batch, low_classifier):
        """high level classifier."""
        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        with tf.variable_scope("low_classifier") as low_scope:
            # [batch_size, 1, html_len, we_dim]
            target_exp = tf.expand_dims(target_batch, 1)

            un_rel = tf.sparse_tensor_to_dense(un_batch)
            un_rel = tf.reshape(un_rel, [FLAGS.batch_size, -1, FLAGS.html_len,
                                         FLAGS.we_dim])
            # concat: unlabeled + target
            un_and_target = tf.concat(1, [target_exp])
            # un_and_target = tf.concat(1, [un_rel, target_exp])

            # call low_classifier to classify relatives
            # all relatives of one target composed of one batch
            # ?? variable scope, init problem of low_classifier ???????
            # [batch_size, num_len(variant) + 1, num_cats]
            # un_and_target = tf.map_fn(low_classifier,
            #                           un_and_target,
            #                           name="map_fn")
            un_and_target = low_classifier(target_batch)
            un_and_target = tf.expand_dims(un_and_target, 1)

            # labeled relatives
        la_rel = tf.sparse_tensor_to_dense(la_batch)
        la_rel = tf.reshape(la_rel, [FLAGS.batch_size, -1, FLAGS.num_cats])

        # concat all inputs for high-level classifier RNN
        # concat_inputs = tf.concat(1, [un_and_target])
        concat_inputs = tf.concat(1, [la_rel, un_and_target])

        # number of pages for each target
        # num_pages = tf.ones([FLAGS.batch_size],
        #                     dtype=tf.int32)
        num_pages = tf.add(
            # tf.add(un_len, la_len),
            la_len,
            tf.ones(
                [FLAGS.batch_size],
                dtype=tf.int32))

        # high-level classifier - RNN
        with tf.variable_scope("dynamic_rnn"):
            cell = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.num_cats)
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=concat_inputs,
                                               sequence_length=num_pages,
                                               dtype=tf.float32)

        return state

    def a_high_classifier(self, page_batch, low_classifier):
        """high level classifier."""
        target_batch, un_batch, un_len, la_batch, la_len = page_batch

        with tf.variable_scope("low_classifier") as low_scope:
            # [batch_size, 1, html_len, we_dim]
            target_exp = tf.expand_dims(target_batch, 1)
            # [batch_size, 1, num_cats]
            target_logits = tf.map_fn(low_classifier,
                                      target_exp,
                                      name="map_fn")

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

        # high-level classifier - RNN
        with tf.variable_scope("dynamic_rnn"):
            cell = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.num_cats)
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=concat_inputs,
                                               sequence_length=num_pages,
                                               dtype=tf.float32)

        return state

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            labels,
            name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')
        # from tensorflow.python.ops import variables
        # added to the collection GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
        tf.add_to_collection('REGULARIZATION_LOSSES', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        total_loss = tf.add_n(
            tf.get_collection('REGULARIZATION_LOSSES'),
            name='total_loss')
        self._add_loss_summaries(total_loss)
        return total_loss

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CNN model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # # Compute the moving average of all individual losses and the total loss.
        # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        # losses = tf.get_collection('losses')
        # loss_averages_op = loss_averages.apply(losses + [total_loss])
        #
        # # Attach a scalar summary to all individual losses and the total loss; do the
        # # same for the averaged version of the losses.
        # for l in losses + [total_loss]:
        #     # Name each loss as '(raw)' and name the moving average version of the loss
        #     # as the original loss name.
        #     tf.scalar_summary(l.op.name + ' (raw)', l)
        #     tf.scalar_summary(l.op.name, loss_averages.average(l))

        losses = tf.get_collection('REGULARIZATION_LOSSES')
        # all_losses = losses + [total_loss]
        all_losses = [total_loss]
        # is it necessary to add all REGULARIZATION_LOSSES ?????
        for l in all_losses:
            tf.scalar_summary(l.op.name, l)

    def training(self, total_loss, global_step):
        """Train CNN model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
            train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = FLAGS.num_train_examples / FLAGS.batch_size
        print("num_batches_per_epoch: {}".format(num_batches_per_epoch))
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)
        print("decay_steps: {}".format(decay_steps))

        # Decay the learning rate exponentially based on the number of steps.
        lr_decay = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                              global_step,
                                              decay_steps,
                                              self.LEARNING_RATE_DECAY_FACTOR,
                                              staircase=True)
        # compare with 0.01 * 0.5^10
        lr = tf.maximum(lr_decay, self.min_lr)
        tf.scalar_summary('learning_rate', lr)

        # optimizer = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        return train_op

    def train_step(self, sess):
        """run one step on one batch trainning examples."""
        step, _, loss_value, top_k = sess.run([self.global_step, self.train_op,
                                               self.loss, self.top_k_op])
        # step, loss_value, top_k = sess.run([self.global_step,
        #                                        self.loss, self.top_k_op])
        # step, l = sess.run([self.global_step, self.logits])
        return step, loss_value, top_k
        # return step, l, 2

    def eval_once(self, sess):
        # it's better to divide exactly with no remainder
        num_iter = int(math.ceil(FLAGS.num_test_examples / FLAGS.batch_size))
        true_count = 0  # counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        eval_step = 0
        while eval_step < num_iter:
            predictions = sess.run([self.top_k_op_eval])
            true_count += np.sum(predictions)
            eval_step += 1

        # compute precision @ 1.
        precision = true_count / total_sample_count
        return precision
