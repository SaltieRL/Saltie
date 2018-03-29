import tensorflow as tf

from bot_code.modelHelpers import tensorflow_feature_creator
from bot_code.models.base_model import BaseModel
from bot_code.conversions.input.input_formatter import InputFormatter


class BaseLayersModel(BaseModel):
    '''
    A base class for any model that is based on tf layers.
    '''


    def __init__(self, session, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__(session, input_dim=input_dim, output_dim=output_dim, **kwargs)

    def _create_model(self, model_input):
        all_layers = [model_input]
        layer_count = 5
        units = 64
        activation = tf.nn.relu
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        dropout_rate = 0.4
        output_activation = tf.nn.tanh

        for i in range(layer_count):
            _layer = tf.layers.dense(all_layers[-1], units, activation=activation,
                                     kernel_regularizer=regularizer)
            _dropout = tf.layers.dropout(_layer, dropout_rate)

            all_layers.append(_layer)
            all_layers.append(_dropout)

        output_layer = tf.layers.dense(all_layers[-1], self.output_dim, activation=output_activation,
                        kernel_regularizer=regularizer)
        self.output_layer = output_layer
        return output_layer

    def create_train_step(self):
        self.labels = tf.placeholder(tf.int64, shape=(None, self.output_dim))
        cost = tf.losses.mean_squared_error(self.labels, self.output_layer)
        loss = tf.reduce_mean(cost, name='xentropy_mean')

        self.train_op = self.optimizer.minimize(loss)