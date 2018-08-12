from examples.base_keras_model import BaseKerasModel
from framework.input_formatter.base_input_formatter import BaseInputFormatter
import tensorflow as tf


class VariationalAutoencoderModel(BaseKerasModel):
    input_dim = 0
    compressed_layer = None

    def __init__(self, encoder=True, decoder=True, compressed_dim=10, hidden_dim=50):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim
        self.decoder = decoder
        self.encoder = encoder

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        self.input_dim = input_placeholder.get_input_state_dimension()[-1]
        self.inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        # super().create_input_layer(input_placeholder)

    def create_hidden_layers(self, input_layer=None):
        if input_layer is None:
            input_layer = self.inputs
        hidden_layer = input_layer

        if self.encoder:
            hidden_layer = self.create_encoder(hidden_layer)

        # adding compressed layer
        hidden_layer = self.compressed_layer = tf.keras.layers.Dense(self.compressed_dim,
                                                                     kernel_regularizer=self.kernel_regularizer,
                                                                     activation=self.activation)(hidden_layer)

        if self.decoder:
            hidden_layer = self.create_decoder(hidden_layer)

        self.hidden_layer = hidden_layer
        return self.hidden_layer

    def create_encoder(self, hidden_layer):
        hidden_layer = tf.keras.layers.Dense(self.hidden_dim, kernel_regularizer=self.kernel_regularizer,
                                             activation=self.activation)(hidden_layer)
        # z mean
        z_mean = tf.keras.layers.Dense(self.compressed_dim, kernel_regularizer=self.kernel_regularizer,
                                       activation=self.activation)(hidden_layer)
        # z log var
        variance = tf.keras.layers.Dense(self.compressed_dim, kernel_regularizer=self.kernel_regularizer,
                                         activation=self.activation)(hidden_layer)
        hidden_layer = tf.keras.layers.Lambda(self.sampling, output_shape=(self.compressed_dim,), name='z')(
            [z_mean, variance])
        return hidden_layer

    def create_decoder(self, hidden_layer):
        hidden_layer = tf.keras.layers.Dense(self.hidden_dim, kernel_regularizer=self.kernel_regularizer,
                                             activation=self.activation)(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(self.input_dim, kernel_regularizer=self.kernel_regularizer,
                                             activation=self.activation)(hidden_layer)
        return hidden_layer

    def sampling(self, args):
        z_mean, variance = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # random normal with default
        epsilon = tf.random_normal(shape=(batch, dim))
        # use z_mean and z_log to sample from the real distribution
        return z_mean + tf.exp(0.5 * variance) * epsilon
