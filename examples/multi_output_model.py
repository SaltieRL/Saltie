import tensorflow as tf
from examples.base_keras_model import BaseKerasModel
from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter
import numpy as np


class MultiOutputKerasModel(BaseKerasModel):

    outputs_list = {'boolean': ['jump', 'boost', 'handbrake'], 'other': [
        'throttle', 'steer', 'pitch', 'yaw', 'roll']}

    def __init__(self, wrapped_model: BaseKerasModel, outputs_list=None):
        super().__init__()
        self.wrapped_model = wrapped_model
        if outputs_list is not None:
            self.outputs_list = outputs_list

    def create_input_layer(self, input_placeholder: BaseInputFormatter):
        self.inputs = self.wrapped_model.create_input_layer(input_placeholder)
        return self.inputs

    def create_hidden_layers(self, input_layer=None):
        return self.wrapped_model.create_hidden_layers(input_layer)

    def create_output_layer(self, output_formatter: BaseOutputFormatter, hidden_layer=None):
        self.outputs = []
        for _output_type, _output_type_list in self.outputs_list.items():
            for output_name in _output_type_list:

                if _output_type == 'boolean':
                    activation = 'sigmoid'
                elif _output_type == 'linear':
                    activation = 'linear'
                else:
                    activation = 'tanh'
                _output = tf.keras.layers.Dense(1, activation=activation, name='o_%s' % output_name)(hidden_layer)
                self.outputs.append(_output)

    def create_loss(self):
        loss = {}
        loss_weights = {}
        for _output_type, _output_type_list in self.outputs_list.items():
            for output_name in _output_type_list:
                loss[
                    'o_%s' % output_name] = 'binary_crossentropy' if _output_type == 'boolean' else 'mean_absolute_error'
                loss_weights['o_%s' %
                             output_name] = 0.5 if _output_type == 'boolean' else 1

        # loss_weights['o_steer'] *= 20
        # loss_weights['o_boost'] *= 10
        # loss_weights['o_throttle'] *= 20
        # loss_weights['o_jump'] *= 20
        # loss_weights['o_pitch'] *= 3
        # loss_weights['o_pitch'] *= 0.001
        # loss_weights['o_yaw'] *= 0.001
        # loss_weights['o_roll'] *= 0.001
        return loss, loss_weights

    def fit(self, x, y, rewards=None, batch_size=1):
        y = np.array_split(y, np.ma.size(y, axis=-1), axis=-1)
        super().fit(x, y, batch_size)

    def write_log(self, callback, names, logs, batch_no, eval=False):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            tag_name = name
            if eval:
                tag_name = 'eval_' + tag_name
            else:
                tag_name = 'train_' + tag_name
            summary_value.tag = tag_name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
