from tensorflow.python.keras import Model

from examples.base_keras_model import BaseKerasModel
from framework.output_formatter.base_output_formatter import BaseOutputFormatter
import tensorflow as tf


class ShotModel(BaseKerasModel):

    def create_output_layer(self, output_formatter: BaseOutputFormatter, hidden_layer=None):
        # sigmoid/tanh all you want on self.model
        if hidden_layer is None:
            hidden_layer = self.hidden_layer
        self.outputs = tf.keras.layers.Dense(output_formatter.get_model_output_dimension()[0],
                                             activation='sigmoid')(hidden_layer)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        return self.outputs
