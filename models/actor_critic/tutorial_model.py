from models import base_model
from models.actor_critic.policy_gradient import PolicyGradient
import tensorflow as tf
import numpy as np


class TutorialModel(PolicyGradient):
    num_split_layers = 7
    gated_layer_index = -1
    split_hidden_layer_variables = None
    split_hidden_layer_name = "split_hidden_layer"
    gated_layer_name = "gated_layer"
    max_gradient = 10.0
    total_loss_divider = 2.0

    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def printParameters(self):
        super().printParameters()
        print('TutorialModel Parameters:')
        print('number of split layers:', self.num_split_layers)
        print('gate layer (not used if < 0):', self.gated_layer_index)

    def load_config_file(self):
        super().load_config_file()
        try:
            self.num_split_layers = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                    'num_split_layers')
        except:
            print('unable to load num_split_layers')
        try:
            self.gated_layer_index = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                            'gated_layer_index')
        except:
            print('unable to load gated_layer_index')

        self.num_split_layers = min(self.num_split_layers, self.num_layers - 2)

    def create_training_op(self, logprobs, labels):
        actor_gradients, actor_loss, actor_reg_loss = self.create_actor_gradients(logprobs, labels)

        tf.summary.scalar("total_reg_loss", actor_reg_loss)

        return self._compute_training_op(actor_gradients, [])

    def create_advantages(self):
        return tf.constant(1.0)

    def fancy_calculate_number_of_ones(self, number):
        """Only use this once it is supported"""
        bitwise1 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(number, 1), tf.constant(0o33333333333))
        bitwise2 = tf.bitwise.bitwise_and(tf.bitwise.right_shift(number, 2), tf.constant(0o11111111111))
        uCount = number - bitwise1 - bitwise2

        bitwise3 = tf.bitwise.bitwise_and(uCount + tf.bitwise.right_shift(uCount, 3), 0o30707070707)
        return tf.mod(bitwise3, 63)

    def normal_calculate_number_of_ones(self, number):
        n = tf.cast(number, tf.int32)

        def body(n, counter):
            counter+= 1
            n = tf.bitwise.bitwise_and(n, n-1)
            return n, counter

        counter = tf.constant(0)
        n, counter = tf.while_loop(lambda n, counter:
                                   tf.not_equal(n, tf.constant(0)),
                                   body, [n, counter], back_prop=False)
        return counter

    def calculate_loss_of_actor(self, logprobs, taken_actions, index):
        cross_entropy_loss, initial_wrongness, __ = super().calculate_loss_of_actor(logprobs, taken_actions, index)
        wrongNess = tf.constant(initial_wrongness)
        if self.action_handler.action_list_names[index] != 'combo':
            wrongNess += tf.cast(tf.abs(tf.cast(self.argmax[index], tf.int32) - taken_actions), tf.float32)
        else:
            # use temporarily
            wrongNess += tf.cast(tf.abs(tf.cast(self.argmax[index], tf.int32) - taken_actions), tf.float32) / 2.0
            #argmax = self.argmax[index]
            #number = tf.bitwise.bitwise_xor(tf.cast(self.argmax[index], tf.int32), taken_actions)
            # result = self.fancy_calculate_number_of_ones(number) # can't use until version 1.5

        return cross_entropy_loss, wrongNess, False

    def create_gated_layer(self, inner_layer, input_state, layer_number, network_size, network_prefix, variable_list=None, scope=None):
        with tf.variable_scope(self.gated_layer_name):
            weight_input = network_prefix + "Winput" + str(layer_number)
            weight_network = network_prefix + "Wnetwork" + str(layer_number)
            weight_decider = network_prefix + "Wdecider" + str(layer_number)

            cut_size = network_size // 2.0

            w_input = tf.get_variable(weight_input, [network_size, cut_size],
                                     initializer=tf.random_normal_initializer())
            w_network = tf.get_variable(weight_network, [network_size, cut_size],
                                      initializer=tf.random_normal_initializer())
            w_decider = tf.get_variable(weight_decider, [network_size, cut_size],
                                        initializer=tf.random_normal_initializer())

            if variable_list is not None:
                variable_list.append(w_network)
                variable_list.append(w_decider)

            decider = tf.nn.sigmoid(tf.matmul(inner_layer, w_decider), name="decider" + str(layer_number))

            left = tf.matmul(input_state, w_input) * decider
            right = tf.matmul(inner_layer, w_network) * (tf.constant(1.0) - decider)

            return left + right, cut_size

    def create_hidden_layers(self, activation_function, input_layer, network_size, network_prefix, variable_list=None,
                             layers_list=[]):
        inner_layer = input_layer
        layer_size = self.network_size
        max_layer = self.num_layers - 2 - self.num_split_layers
        for i in range(0, max_layer):
            if i == self.gated_layer_index:
                inner_layer, layer_size = self.create_gated_layer(inner_layer, input_layer, i + 2, layer_size,
                                                                  network_prefix,
                                                                  variable_list=variable_list)
            else:
                with tf.variable_scope(self.hidden_layer_name):
                    inner_layer, layer_size = self.create_layer(tf.nn.relu6, inner_layer, i + 2, layer_size,
                                                                self.network_size,
                                                                network_prefix, variable_list=variable_list)
        return inner_layer, layer_size

    def create_last_layer(self, activation_function, inner_layer, network_size, num_actions, network_prefix,
                          last_layer_list=None, layers_list=[]):
        with tf.variable_scope(self.split_hidden_layer_name):
            inner_layers, layer_size = self.create_split_layers(tf.nn.relu6, inner_layer, network_size,
                                                                self.num_split_layers,
                                                                network_prefix,
                                                                variable_list=last_layer_list)

        for layer in inner_layers:
            layers_list.append(layer)
        output_layers = inner_layers[len(inner_layers) - 1]
        return super().create_last_layer(activation_function, output_layers, layer_size, num_actions, network_prefix,
                                         last_layer_list, layers_list=layers_list)

    def create_split_layers(self, activation_function, inner_layer, network_size,
                            num_split_layers, network_prefix, variable_list=None):

        cut_size = self.network_size // 3
        total_layers = []
        previous_layer = []
        last_sizes = []
        step_size = (network_size - cut_size) // num_split_layers
        for i in reversed(np.arange(cut_size, network_size, step_size)):
            layer_size = []
            for j in range(self.action_handler.get_number_actions()):
                layer_size.append(i)
            last_sizes.append(layer_size)
        layer_size = []
        last_layer_size = last_sizes[len(last_sizes) - 1]
        for j in range(self.action_handler.get_number_actions()):
            previous_layer.append(inner_layer)
            layer_size.append(network_size)
        # needs to be one more longer then the number of layers
        last_sizes.insert(0, layer_size)
        for i in range(0, num_split_layers):
            split_layers = []
            for j, item in enumerate(self.action_handler.get_action_sizes()):
                name = str(i)
                with tf.variable_scope(str(self.action_handler.action_list_names[j])):
                    inner_layer, last_layer_size = self.create_layer(activation_function, previous_layer[j], 'split' + name,
                                                       last_sizes[i][j], last_sizes[i + 1][j], network_prefix,
                                                       variable_list=variable_list[j])
                    split_layers.append(inner_layer)
            previous_layer = split_layers
            total_layers.append(split_layers)
        return total_layers, last_layer_size

    def get_model_name(self):
        return 'tutorial_bot' + ('_split' if self.action_handler.is_split_mode else '')

    def create_savers(self):
        super().create_savers()
        # self._create_layer_saver('actor_network', self.split_hidden_layer_name)
        self._create_layer_saver('actor_network', self.gated_layer_name)

    def _create_last_row_saver(self, network_name):
        super()._create_last_row_saver(network_name)
        # create the hidden row savers
        split_las_layer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=network_name + '/' + self.split_hidden_layer_name + '.*')
        reshaped_list = np.reshape(np.array(split_las_layer), [-1, self.action_handler.get_number_actions(), 2])
        for i in range(len(reshaped_list)):
            for j in range(len(reshaped_list[i])):
                self._create_layer_saver(network_name, self.split_hidden_layer_name + '_' + str(i),
                                         extra_info=self.action_handler.action_list_names[j],
                                         variable_list=reshaped_list[i][j].tolist())

    def add_histograms(self, gradients):
        # summarize gradients
        for grad, var in gradients:
            tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + '/gradients', grad)
