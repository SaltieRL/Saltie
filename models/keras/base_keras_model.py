from conversions.input.simple_input_formatter import SimpleInputFormatter
from models.base_model import BaseModel
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LeakyReLU, PReLU
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.utils import plot_model


class BaseKerasModel(BaseModel):

    shared_hidden_layers = 0
    split_hidden_layers = 0
    model_activation = None
    kernel_regularizer = None
    loss_weights = None
    loss = None

    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=None, summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def printParameters(self):
        super().printParameters()

    def _create_variables(self):
        pass

    def add_input_formatter(self, team, index):
        self.input_formatter = SimpleInputFormatter(team, index)

    def sample_action(self, input_state):
        return self.model.predict(input_state)

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

        # adam = optimizers.Adam(lr=0.01)

    def get_input(self, model_input=None):
        return Input(shape=(self.input_formatter.get_state_dim(),))

    def _create_model(self, model_input):
            #def generate_model(self, input_dim, outputs=1, shared_hidden_layers=0, nodes=256, extra_hidden_layers=6, extra_hidden_layer_nodes=128):
        #"""Generates and returns Keras model given input dim, outputs, hidden_layers, and nodes"""

        x = model_input
        for hidden_layer_i in range(1, self.shared_hidden_layers + 1):
            x = Dense(self.network_size, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer, name='hidden_layer_%s' %
                      hidden_layer_i)(x)
            x = Dropout(0.4)(x)

        shared_output = x
        outputs = []

        extra_hidden_layer_nodes = self.network_size / self.action_handler.get_number_actions()
        loss = {}
        for i, control in enumerate(self.action_handler.control_names):
            output_size = self.action_handler.get_action_sizes()[i]
            x = shared_output
            for hidden_layer_i in range(1, self.split_hidden_layers + 1):
                x = Dense(extra_hidden_layer_nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                          name='hidden_layer_%s_%s' % (control, hidden_layer_i))(x)
                x = Dropout(0.4)(x)

            if self.action_handler.is_classification(i):
                activation = 'sigmoid'
                loss = 'categorical_crossentropy'
            else:
                activation = 'tanh'
                loss = 'mean_absolute_error'
            _output = Dense(output_size, activation=activation,
                            name='o_%s' % control)(x)
            outputs.append(_output)
            loss['o_%s' % control] = loss

        self.loss = loss

        self.model = Model(inputs=model_input, outputs=outputs)

        return None

    def initialize_model(self):
        if self.loss_weights is not None:
            self.model.compile(optimizer='adam', loss=self.loss, loss_weights=self.loss_weights)
        else:
            self.model.compile()
        super().initialize_model()

    def _initialize_variables(self):
        super()._initialize_variables()

    def run_train_step(self, calculate_summaries, input_states, actions):
        super().run_train_step(calculate_summaries, input_states, actions)

    def _add_summary_writer(self):
        super()._add_summary_writer()

    def load_config_file(self):
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
