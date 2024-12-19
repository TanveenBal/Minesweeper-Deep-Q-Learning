import tensorflow as tf
from keras.callbacks import TensorBoard

# Use with TensorFlow version 2+
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self._train_dir = self.log_dir
        self._train_step = 0
        self._should_write_train_graph = False

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overridden, saves logs with our step number
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.update_stats(**logs)
        self.step += 1  # Increment step for the next epoch

    # This should now be logging during training to track the step properly
    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            self.update_stats(**logs)
        self.step += 1  # Increment step for the next batch

    # Overridden, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()  # Ensure the data is written immediately
