from bot_code.models.actor_critic.policy_gradient import PolicyGradient
from bot_code.models.base_agent_model import BaseAgentModel
import tensorflow as tf


class RnnAC(PolicyGradient):

    variational_recurrent = True
    num_steps = 10
    num_cells = 1
    layers_per_cell = 1

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None,
                 discount_factor=0.99,  # discount future rewards
                 ):

        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file, discount_factor)

        #save the entire rnn network
        self.saver = tf.train.Saver(tf.trainable_variables())

    def load_config_file(self):
        super().load_config_file()
        try:
            self.num_cells = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                     'num_cells')
        except Exception as e:
            print('using default cell number')

        try:
            self.layers_per_cell = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                 'layers_per_cell')
        except Exception as e:
            print('using default layers_per_cell')

    def actor_network(self, input_states):
        # define policy neural network
        actor_prefix = 'actor'
        layer1, _ = self.create_layer(tf.nn.relu6, input_states, 1, self.state_feature_dim, self.network_size, actor_prefix)
        inner_layer = layer1
        state = None
        for i in range(0, self.num_cells):
            lstms = []
            for j in range(0, self.layers_per_cell):
                lstms.append(self.create_layer_rnn(self.network_size))
            inner_layer, state = self.create_cell_rnn(inner_layer, lstms, state, i + 1, actor_prefix)

        with tf.variable_scope("last_layer"):
            output_layer = self.create_last_layer(tf.nn.sigmoid, inner_layer, self.network_size,
                                                  self.num_actions, actor_prefix)

        return output_layer

    def create_layer_rnn(self, input_size):
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell(input_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.AUTO_REUSE)
        return basic_cell

    def create_cell_rnn(self, input, layers, initial_state, layer_number, network_prefix):
        rnnName = network_prefix + "rnnCell" + str(layer_number)
        if self.is_training:
            layers = [tf.contrib.rnn.DropoutWrapper(
                layer,
                output_keep_prob=self.keep_prob,
                variational_recurrent=self.variational_recurrent,
                dtype=tf.float32) for layer in layers]

        cell = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)

        if self.is_training:
            batch_size = self.mini_batch_size
        else:
            batch_size = 1

        if initial_state is None:
            state = cell.zero_state(batch_size, dtype=tf.float32)
        else:
            state = initial_state

        if self.is_training:
            cell_output = input
            with tf.variable_scope(rnnName):
                for time_step in range(self.num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(cell_output, state)
        else:
            cell_output, state = cell(input, state)

        final_state = state

        return cell_output, final_state

    def get_model_name(self):
        return 'RNN-' + super().get_model_name()
