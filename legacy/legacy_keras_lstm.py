from framework.input_formatter.base_input_formatter import BaseInputFormatter
from legacy.legacy_keras_model import LegacyKerasModel
import tensorflow as tf


class LegacyKerasLstm(LegacyKerasModel):
    lstm_state = None
    prediction_mode = False

    def __init__(self, prediction_mode=False):
        super().__init__()
        self.prediction_mode = prediction_mode

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        """Creates keras model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_placeholder.get_input_state_dimension()))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        lstm = tf.keras.layers.LSTM(units=512, kernel_regularizer=self.kernel_regularizer, recurrent_dropout=0.1,
                                    return_sequences=True, stateful=self.prediction_mode)
        model.add(lstm)
        super().create_hidden_layers()
