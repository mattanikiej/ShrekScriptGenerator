from model import Model
import tensorflow as tf
import os
from one_step_model import OneStep


class Train:
    """
    Used to run the training on the model
    """

    def __init__(self, epochs=50):
        self.model_ = Model()
        self.model_.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        self.model_.fit(self.model_.training_data_, epochs=epochs, verbose=2, callbacks=[checkpoint_callback])

    def get_one_step(self):
        """
        Returns generator of one step model
        :return: generator for the script
        """
        return OneStep(self.model_, self.model_.invert_ids_, self.model_.tokens_)
