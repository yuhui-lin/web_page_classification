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
