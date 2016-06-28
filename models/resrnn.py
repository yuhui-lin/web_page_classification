"""CNN model class"""
import tensorflow as tf
# import model
import models.resnn

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS


class ResRNN(models.resnn.ResNN):
    """Residual neural network model.
    classify web page only based on target html."""

    def inference(self, page_batch):
        """Build the ResRNN model.
        Args:
            page_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        self.activation = tf.nn.relu
        self.norm_decay = 0.99

        return self.high_classifier(page_batch, self.resnn)
