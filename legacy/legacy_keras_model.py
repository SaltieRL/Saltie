from random import random

import tensorflow as tf

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.model.base_model import BaseModel
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LegacyKerasModel(BaseModel):

    model = None
    tensorboard = None
    train_names = ['train_loss', 'train_mae']
    val_names = ['val_loss', 'val_mae']
    counter = 0

    def __init__(self, use_default_dense=True, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)):
        if use_default_dense:
            self.activation = activation
            self.kernel_regularizer = kernel_regularizer

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        """Creates keras model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[input_placeholder.get_input_state_dimension()]))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation))

    def create_output_layer(self, output_formatter: BaseOutputFormatter):
        # sigmoid/tanh all you want on self.model
        self.model.add(tf.keras.layers.Dense(output_formatter.get_model_output_dimension(), activation='tanh'))
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
        self.model.compile(tf.keras.optimizers.Nadam(), loss='mse', metrics=['mse'])
        log_name = './logs/' + str(int(random() * 1000))
        print("log_name", log_name)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_name,
                                                          histogram_freq=1,
                                                          write_images=False,
                                                          batch_size=1000,
                                                          )
        self.tensorboard.set_model(self.model)

    def fit(self, x, y):
        if self.counter % 200 == 0:
            logs = self.model.evaluate(x, y)
            self.write_log(self.tensorboard, self.train_names, logs, self.counter)
        else:
            logs = self.model.train_on_batch(x, y)
            self.write_log(self.tensorboard, self.train_names, logs, self.counter)
        self.counter += 1

    def predict(self, arr):
        return self.model.predict(arr)
