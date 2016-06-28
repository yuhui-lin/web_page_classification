"""CNN model class"""
import tensorflow as tf
# import model
import models.cnn

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class CRNN(models.cnn.CNN):
    """convolutional neural network model.
    classify web page only based on target html."""

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

        return self.high_classifier(page_batch, self.cnn)
