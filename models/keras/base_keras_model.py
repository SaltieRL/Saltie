from models.base_model import BaseModel


class BaseKerasModel(BaseModel):

    shared_hidden_layers = 0
    split_hidden_layers = 0
    model_activation = None
    kernel_regularizer = None

    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=None, summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def printParameters(self):
        super().printParameters()

    def _create_variables(self):
        pass

    def sample_action(self, input_state):
        return super().sample_action(input_state)

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        return super().create_copy_training_model(model_input, taken_actions)

    def get_input(self, model_input=None):
        # given maybe input return keras version  for your model
        # note that the super class uses a tensorflow placeholder

        inputs = Input(shape=(input_dim,))
        return super().get_input(model_input)

    def _create_model(self, model_input):
            #def generate_model(self, input_dim, outputs=1, shared_hidden_layers=0, nodes=256, extra_hidden_layers=6, extra_hidden_layer_nodes=128):
        #"""Generates and returns Keras model given input dim, outputs, hidden_layers, and nodes"""

        x = model_input
        for hidden_layer_i in range(1, self.shared_hidden_layers + 1):
            x = Dense(self.network_size, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer, name='hidden_layer_%s' %
                      hidden_layer_i)(x)
            x = Dropout(0.4)(x)

        shared_output = x

        outputs_list = {'boolean': ['jump', 'boost', 'handbrake'], 'other': [
            'throttle', 'steer', 'pitch', 'yaw', 'roll']}
        outputs = []
        for _output_type, _output_type_list in outputs_list.items():
            for output_name in _output_type_list:
                x = shared_output
                for hidden_layer_i in range(1, extra_hidden_layers + 1):
                    x = Dense(extra_hidden_layer_nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                              name='hidden_layer_%s_%s' % (output_name, hidden_layer_i))(x)
                    x = Dropout(0.4)(x)

                if _output_type == 'boolean':
                    activation = 'sigmoid'
                else:
                    activation = 'tanh'
                _output = Dense(1, activation=activation,
                                name='o_%s' % output_name)(x)
                outputs.append(_output)

        extra_hidden_layer_nodes = self.network_size / self.action_handler.get_number_actions()
        loss = {}
        for i, control in enumerate(self.action_handler.control_names):
            output_size = self.action_handler.get_action_sizes()[i]
            x = shared_output
            for hidden_layer_i in range(1, self.split_hidden_layers + 1):
                x = Dense(extra_hidden_layer_nodes, activation=self.model_activation, kernel_regularizer=self.kernel_regularizer,
                          name='hidden_layer_%s_%s' % (control, hidden_layer_i))(x)
                if self.is_training:
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


        model = Model(inputs=model_input, outputs=outputs)

        loss = {}
        loss_weights = {}
        for _output_type, _output_type_list in outputs_list.items():
            for output_name in _output_type_list:
                loss[
                    'o_%s' % output_name] = 'binary_crossentropy' if _output_type == 'boolean' else 'mean_absolute_error'
                loss_weights['o_%s' %
                             output_name] = 0.01 if _output_type == 'boolean' else 0.1

        loss_weights['o_steer'] *= 20
        loss_weights['o_boost'] *= 10
        loss_weights['o_throttle'] *= 20
        loss_weights['o_jump'] *= 20
        # loss_weights['o_pitch'] *= 3
        # loss_weights['o_pitch'] *= 0.001
        # loss_weights['o_yaw'] *= 0.001
        # loss_weights['o_roll'] *= 0.001

        # adam = optimizers.Adam(lr=0.01)
        model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights)

        return model
        return super()._create_model(model_input)

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
