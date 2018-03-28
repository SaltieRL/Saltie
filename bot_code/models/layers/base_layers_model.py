import tensorflow as tf

from bot_code.modelHelpers import tensorflow_feature_creator
from bot_code.models.base_model import BaseModel
from bot_code.conversions.input.input_formatter import InputFormatter


class BaseLayersModel(BaseModel):
    '''
    A base class for any model that is based on tf layers.
    '''

    def __init__(self, session, input_dim, output_dim, **kwargs):
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

        output_layer = tf.layers.dense(all_layers[-1], units, activation=output_activation,
                        kernel_regularizer=regularizer)

        return output_layer