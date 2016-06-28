import tensorflow as tf
import numpy as np


def _testDynamic():
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    with tf.Session(graph=tf.Graph()) as sess:
        concat_inputs = tf.placeholder(
            tf.float32,
            shape=(time_steps, batch_size, input_size))

        cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)

        with tf.variable_scope("dynamic_scope"):
            outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
                cell,
                inputs=concat_inputs,
                sequence_length=sequence_length,
                time_major=True,
                dtype=tf.float32)

        feeds = {concat_inputs: input_values}

        # Initialize
        tf.initialize_all_variables().run(feed_dict=feeds)

        sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)
        print("end of sess")


_testDynamic()


def testDynamicRNNWithTupleStates():
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    with tf.Session(graph=tf.Graph()) as sess:
        initializer = tf.random_uniform_initializer(-0.01,
                                                    0.01,
                                                    seed=self._seed)
        inputs = max_length * [
            tf.placeholder(tf.float32,
                           shape=(None, input_size))
        ]
        inputs_c = tf.pack(inputs)
        cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                       use_peepholes=True,
                                       num_proj=num_proj,
                                       initializer=initializer,
                                       state_is_tuple=True)
        outputs_static, state_static = tf.nn.rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=sequence_length)
        tf.get_variable_scope().reuse_variables()
        outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
            cell,
            inputs_c,
            dtype=tf.float32,
            time_major=True,
            sequence_length=sequence_length)
        self.assertTrue(isinstance(state_static,
                                   tf.nn.rnn_cell.LSTMStateTuple))
        self.assertTrue(isinstance(state_dynamic,
                                   tf.nn.rnn_cell.LSTMStateTuple))
        self.assertEqual(state_static[0], state_static.c)
        self.assertEqual(state_static[1], state_static.h)
        self.assertEqual(state_dynamic[0], state_dynamic.c)
        self.assertEqual(state_dynamic[1], state_dynamic.h)

        tf.initialize_all_variables().run()

        input_value = np.random.randn(batch_size, input_size)
        outputs_static_v = sess.run(outputs_static,
                                    feed_dict={inputs[0]: input_value})
        outputs_dynamic_v = sess.run(outputs_dynamic,
                                     feed_dict={inputs[0]: input_value})
        self.assertAllEqual(outputs_static_v, outputs_dynamic_v)

        state_static_v = sess.run(state_static,
                                  feed_dict={inputs[0]: input_value})
        state_dynamic_v = sess.run(state_dynamic,
                                   feed_dict={inputs[0]: input_value})
        self.assertAllEqual(
            np.hstack(state_static_v), np.hstack(state_dynamic_v))
