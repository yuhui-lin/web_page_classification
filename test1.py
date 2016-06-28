import numpy as np
import tensorflow as tf


class LSTMTest(tf.test.TestCase):
    def setUp(self):
        self._seed = 23489
        np.random.seed(self._seed)

    def testDynamicRNNWithTupleStates(self):
        num_units = 3
        input_size = 5
        batch_size = 2
        num_proj = 4
        max_length = 8
        sequence_length = [4, 6]
        with self.test_session(graph=tf.Graph()) as sess:
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
            print("done~")
