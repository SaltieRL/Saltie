from conversions.input.simple_input_formatter import SimpleInputFormatter
from models.base_model import BaseModel, MODEL_CONFIGURATION_HEADER
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, PReLU
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, Callback, TensorBoard
# from keras.utils import plot_model
import numpy as np


class BaseKerasModel(BaseModel):

    shared_hidden_layers = 0
    split_hidden_layers = 0
    model_activation = None
    kernel_regularizer = None
    loss_weights = None
    loss = None

    def printParameters(self):
        super().printParameters()

    def _create_variables(self):
        pass

    def add_input_formatter(self, team, index):
        self.input_formatter = SimpleInputFormatter(team, index)

    def sample_action(self, input_state):
        relative_positions = input_state[:, 13:16] - input_state[:, 0:3]
        rotations = input_state[:, 3:6]
        unrotated_positions = self.unrotate_positions(relative_positions, rotations)

        input_state = np.column_stack((input_state, unrotated_positions))
        outputs = self.model.predict(input_state)
        outputs = [output[0] for output in outputs]
        return outputs

    def unrotate_positions(self, relative_positions, rotations):
        new_positions = relative_positions

        # YAW
        yaws = rotations[:, 1]
        yaws = -yaws / 32768. * np.pi

        new_positions[:, 0], new_positions[:, 1] = new_positions[:, 0] * np.cos(yaws) - new_positions[:, 1] * np.sin(yaws), new_positions[:, 0] * np.sin(yaws) + new_positions[:, 1] * np.cos(yaws)

        # PITCH

        pitchs = rotations[:, 0]
        pitchs = pitchs / 32768. * np.pi

        new_positions[:, 2], new_positions[:, 0] = new_positions[:, 2] * np.cos(pitchs) - new_positions[:, 0] * np.sin(pitchs), new_positions[:, 2] * np.sin(pitchs) + new_positions[:, 0] * np.cos(pitchs)

        # ROLL

        rolls = rotations[:, 2]
        rolls = rolls / 32768. * np.pi

        new_positions[:, 1], new_positions[:, 2] = new_positions[:, 1] * np.cos(rolls) - new_positions[:, 2] * np.sin(rolls), new_positions[:, 1] * np.sin(rolls) + new_positions[:, 2] * np.cos(rolls)

        return new_positions

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        loss_weights = {}
        for i, control in enumerate(self.action_handler.control_names):
            is_classification = self.action_handler.is_classification(i)
            loss_weights['o_%s' %
                         control] = 0.01 if is_classification else 0.1

        loss_weights['o_steer'] *= 20
        loss_weights['o_boost'] *= 10
        loss_weights['o_throttle'] *= 20
        loss_weights['o_jump'] *= 20
        self.loss_weights = loss_weights
        # loss_weights['o_pitch'] *= 3
        # loss_weights['o_pitch'] *= 0.001
        # loss_weights['o_yaw'] *= 0.001
        # loss_weights['o_roll'] *= 0.001

    def get_input(self, model_input=None):
        return Input(shape=(self.state_dim + 3,))

    def _create_model(self, model_input):
        """Generates the Keras model"""

        x = model_input
        for hidden_layer_i in range(1, self.shared_hidden_layers + 1):
            x = Dense(self.network_size, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer, name='hidden_layer_%s' %
                      hidden_layer_i)(x)
            x = Dropout(0.4)(x)

        shared_output = x
        outputs = []

        extra_hidden_layer_nodes = self.network_size / self.action_handler.get_number_actions()
        loss = {}
        action_sizes = self.action_handler.get_action_sizes()
        for i, control in enumerate(self.action_handler.action_list_names):
            output_size = action_sizes[i]
            x = shared_output
            for hidden_layer_i in range(1, self.split_hidden_layers + 1):
                x = Dense(extra_hidden_layer_nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                          name='hidden_layer_%s_%s' % (control, hidden_layer_i))(x)
                x = Dropout(0.4)(x)

            if self.action_handler.is_classification(i):
                activation = 'sigmoid'
                loss_name = 'categorical_crossentropy'
            else:
                activation = 'tanh'
                loss_name = 'mean_absolute_error'
            _output = Dense(output_size, activation=activation,
                            name='o_%s' % control)(x)
            outputs.append(_output)
            loss['o_%s' % control] = loss_name

        self.loss = loss

        self.model = Model(inputs=model_input, outputs=outputs)

        return None

    def initialize_model(self):
        self.model.compile(optimizer='adam', loss=self.loss, loss_weights=self.loss_weights)
        super().initialize_model()

    def _initialize_variables(self):
        super()._initialize_variables()

    def run_train_step(self, calculate_summaries, input_states, actions):
        super().run_train_step(calculate_summaries, input_states, actions)

    def _add_summary_writer(self):
        super()._add_summary_writer()

    def load_config_file(self):
        """Loads a config file.  The config file is stored in self.config_file"""
        try:
            self.model_file = self.config_file.get(MODEL_CONFIGURATION_HEADER, 'model_directory')
        except Exception as e:
            print('model directory is not in config', e)
        super().load_config_file()

    def add_saver(self, name, variable_list):
        super().add_saver(name, variable_list)

    def create_savers(self):
        super().create_savers()

    def _save_model(self, session, saver, file_path, global_step):
        super()._save_model(session, saver, file_path, global_step)

    def _load_model(self, session, saver, path):
        super()._load_model(session, saver, path)

    def create_model_hash(self):
        return super().create_model_hash()

    def get_model_name(self):
        return 'keras'

    def get_input_placeholder(self):
        return None
