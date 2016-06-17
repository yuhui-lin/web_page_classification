"""neural network model."""
import re

import tensorflow as tf

import inputs

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
# parameters applied for both train.py and eval.py will be kept here.
# Basic model parameters.
tf.app.flags.DEFINE_integer("batch_size", 128, "mini Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_layers", 1, "word vector dimension")
tf.app.flags.DEFINE_integer("hidden_layers", 50, "word vector dimension")
tf.app.flags.DEFINE_integer("num_local", 1024, "word vector dimension")

#########################################
# global variables
#########################################
FEATURE_NUM = 256

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


#########################################
# functions
#########################################
class Model(object):
    """super class for neural network models."""

    def __init__(self, is_train=False):
        self.num_cats = FLAGS.num_cats

        if is_train:
            # build training graph
            self.dropout = FLAGS.dropout_keep_prob

            self.global_step = tf.Variable(0, trainable=False)

            # get input data
            # sequences, labels = model.inputs_train()
            # target_batch, unlabeled_batch, labeled_batch, label_batch = model.inputs_train()
            page_batch, label_batch = self.inputs_train()
            # logits = model.get_embedding(sequences)

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
            self.dropout = 1

            page_batch_eval, label_batch_eval = self.inputs_eval()
            self.logits_eval = self.inference(page_batch_eval)
            # Calculate predictions.
            self.top_k_op_eval = tf.nn.in_top_k(self.logits_eval,
                                                label_batch_eval, 1)
            tf.scalar_summary(
                "accuracy_eval",
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
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

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

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
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
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
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
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CNN model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op

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
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr_decay = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                              global_step,
                                              decay_steps,
                                              LEARNING_RATE_DECAY_FACTOR,
                                              staircase=True)
        # compare with 0.01 * 0.5^10
        lr = tf.maximum(lr_decay, 0.000009765625)
        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            # opt = tf.train.GradientDescentOptimizer(lr)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
        #                                                       global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train_step(self, sess):
        """run one step on one batch trainning examples."""
        _, loss_value, top_k = sess.run([self.train_op, self.loss,
                                         self.top_k_op])
        return loss_value, top_k

    def eval_step(self, sess):
        pass
