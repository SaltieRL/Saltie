from random import random

import os
import tensorflow as tf

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.model.base_model import BaseModel
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class ExampleSequentialKerasModel(BaseModel):

    model = None
    tensorboard = None
    train_names = ['train_loss', 'train_mse', 'train_mae']
    val_names = ['val_loss', 'val_mse', 'val_mae']
    counter = 0

    def __init__(self, use_default_dense=True, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)):
        super().__init__()
        if use_default_dense:
            self.activation = activation
            self.kernel_regularizer = kernel_regularizer

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        """Creates keras model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_placeholder.get_input_state_dimension()))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, kernel_regularizer=self.kernel_regularizer, activation=self.activation))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(64, kernel_regularizer=self.kernel_regularizer, activation=self.activation))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation))
        model.add(tf.keras.layers.Dropout(0.1))

    def create_output_layer(self, output_formatter: BaseOutputFormatter):
        # sigmoid/tanh all you want on self.model
        self.model.add(tf.keras.layers.Dense(output_formatter.get_model_output_dimension()[0], activation='tanh'))
        return self.model.layers[-1].output

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def finalize_model(self):
        self.model.compile(tf.keras.optimizers.Nadam(lr=0.001), loss='mean_absolute_error',
                           metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])
        log_name = './logs/' + str(int(random() * 1000))
        self.logger.info("log_name: " + log_name)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_name,
                                                          histogram_freq=1,
                                                          write_images=False,
                                                          batch_size=1000,
                                                          )
        self.tensorboard.set_model(self.model)
        self.logger.info("Model has been finalized")

    def fit(self, x, y, rewards=None, batch_size=1):
        if self.counter % 200 == 0:
            logs = self.model.evaluate(x, y, batch_size=batch_size, verbose=1)
            self.write_log(self.tensorboard, self.val_names, logs, self.counter)
            print('step:', self.counter)
        else:
            logs = self.model.train_on_batch(x, y)
            self.write_log(self.tensorboard, self.train_names, logs, self.counter)
        self.counter += 1

    def predict(self, arr):
        return self.model.predict(arr)

    def save(self, file_path):
        self.model.save_weights(filepath=file_path, overwrite=True)

    def load(self, file_path):
        path = os.path.abspath(file_path)
        self.model.load_weights(filepath=os.path.abspath(file_path))
