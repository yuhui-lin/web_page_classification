"""neural network model."""
import re

import tensorflow as tf
from tensorflow.python.ops import array_ops as array_ops_

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
def _activation_summary(x):
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


def _variable_on_cpu(name, shape, initializer):
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


def _variable_with_weight_decay(name, shape, stddev, wd):
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
    var = _variable_on_cpu(name,
                           shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs_train():
    """Construct input examples for training process.
    Returns:
        sequences: 4D tensor of [batch_size, 1, input_length, alphabet_length] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    return inputs.inputs("train",
                         FLAGS.batch_size,
                         FLAGS.num_epochs,
                         min_shuffle=1000)


def inputs_eval():
    """Construct input examples for evaluations.
    similar to inputs_train
    """
    # don't shuffle
    return inputs.inputs("test", FLAGS.batch_size, None, min_shuffle=1)


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.
    Args:
        input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
        lengths:   A tensor of dimension batch_size, containing lengths for each
                   sequence in the batch. If "None" is specified, simply reverses
                   the list.
    Returns:
        time-reversed sequence
    """
    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))

    # Join into (time, batch_size, depth)
    s_joined = array_ops_.pack(input_seq)

    # Reverse along dimension 0
    s_reversed = array_ops_.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops_.unpack(s_reversed)
    return result


def inference(page_batch):
    """Build the RCNN model.
    Args:
        sequences: Sequences returned from inputs_train() or inputs_eval.
    Returns:
        Logits.
    """
    target_batch, unlabeled_batch, labeled_batch = page_batch
    filter_sizes = [3, 4, 5]
    num_filters = len(filter_sizes)
    dropout_keep_prob = 0.5
    NUM_CLASSES = 5
    sequence_length = FLAGS.html_len
    # [batch_size, html_len, 1, we_dim]
    embedded_chars_expanded = tf.expand_dims(target_batch, 2)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, 1, FLAGS.we_dim, num_filters]
            W = tf.Variable(
                tf.truncated_normal(filter_shape,
                                    stddev=0.1),
                name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
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
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # softmax, i.e. softmax(WX + b)
    with tf.name_scope('softmax_linear'):
        W = tf.get_variable("W",
                            shape=[num_filters_total, NUM_CLASSES],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
        softmax_linear = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

    return softmax_linear


def loss(logits, labels):
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
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
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


def training(total_loss, global_step):
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
    loss_averages_op = _add_loss_summaries(total_loss)

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
