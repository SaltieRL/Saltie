from tensorflow.python.keras.layers import Input

from examples.base_keras_model import BaseKerasModel
from framework.input_formatter.base_input_formatter import BaseInputFormatter
import tensorflow as tf


class ExampleLSTMModel(BaseKerasModel):
    lstm_state = None
    prediction_mode = False

    def __init__(self, prediction_mode=False):
        super().__init__()
        self.prediction_mode = prediction_mode

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        """Creates keras model"""
        if self.prediction_mode:
            shape = [1] + input_placeholder.get_input_state_dimension()
            self.inputs = Input(batch_input_shape=shape)
        else:
            self.inputs = Input(shape=input_placeholder.get_input_state_dimension())
        return self.inputs

    def create_hidden_layers(self, input_layer=None):
        if input_layer is None:
            input_layer = self.inputs
        lstm = tf.keras.layers.LSTM(units=512, kernel_regularizer=self.kernel_regularizer, recurrent_dropout=0.1,
                                    return_sequences=True, stateful=self.prediction_mode)
        self.hidden_layer = lstm(input_layer)
        lstm = tf.keras.layers.LSTM(units=128, kernel_regularizer=self.kernel_regularizer, recurrent_dropout=0.1,
                                    return_sequences=True, stateful=self.prediction_mode)
        self.hidden_layer = lstm(self.hidden_layer)
        return self.hidden_layer

    def predict(self, arr):
        return self.model.predict(arr)
