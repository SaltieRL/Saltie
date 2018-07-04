import tensorflow as tf

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.model.base_model import BaseModel
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LegacyKerasModel(BaseModel):

    model = None

    def __init__(self, use_default_dense=True, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)):
        if use_default_dense:
            self.activation = activation
            self.kernel_regularizer = kernel_regularizer

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        """Creates keras model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[None, input_placeholder.get_input_state_dimension()]))
        self.model = model

    def create_hidden_layers(self):
        model = self.model
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=self.kernel_regularizer, activation=self.activation))

    def create_output_layer(self, output_formatter: BaseOutputFormatter):
        # sigmoid/tanh all you want on self.model
        self.model.add(tf.keras.layers.Dense(output_formatter.get_model_output_dimension(), activation='tanh'))
        return self.model.layers[-1].output

    def finalize_model(self):
        self.model.compile(tf.keras.optimizers.Nadam())

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=1000)

    def predict(self, arr):
        return self.model.predict(arr)
