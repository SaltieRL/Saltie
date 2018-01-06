from models.actor_critic.policy_gradient import PolicyGradient
import tensorflow as tf


class TutorialModel(PolicyGradient):
    num_split_layers = 1
    split_hidden_layer_name = "split_hidden_layer"
    gated_layer_name = "gated_layer"

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)

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
            argmax = self.argmax[index]
            number = tf.bitwise.bitwise_xor(tf.cast(self.argmax[index], tf.int32), taken_actions)
            # result = self.fancy_calculate_number_of_ones(number) # can't use until version 1.5
            result = tf.map_fn(self.normal_calculate_number_of_ones, number)
            wrongNess += tf.cast(result, tf.float32)

        return cross_entropy_loss, wrongNess, False

    def create_gated_layer(self, inner_layer, layer_number, network_size, network_prefix, variable_list=None, scope=None):
        with tf.variable_scope(self.gated_layer_name):
            weight_left = network_prefix + "Wleft" + str(layer_number)
            weight_right = network_prefix + "Wright" + str(layer_number)
            weight_decider = network_prefix + "Wdecider" + str(layer_number)
            w_left = tf.get_variable(weight_left, [network_size, network_size / 2],
                                     initializer=tf.random_normal_initializer())
            w_right = tf.get_variable(weight_right, [network_size, network_size / 2],
                                      initializer=tf.random_normal_initializer())
            w_decider = tf.get_variable(weight_decider, [network_size, network_size / 2],
                                        initializer=tf.random_normal_initializer())

            if variable_list is not None:
                variable_list.append(w_left)
                variable_list.append(w_right)
                variable_list.append(w_decider)

            decider = tf.nn.sigmoid(tf.matmul(inner_layer, w_decider), name="decider" + str(layer_number))

            left = tf.matmul(inner_layer, w_left) * decider
            right = tf.matmul(inner_layer, w_right) * (tf.constant(1.0) - decider)

            return left + right, network_size / 2.0

    def create_hidden_layers(self, activation_function, input_layer, network_size, network_prefix,
                             variable_list=None):
        inner_layer = input_layer
        print('num layers', self.num_layers)
        layer_size = self.network_size
        max_layer = self.num_layers - 2 - self.num_split_layers
        for i in range(0, max_layer):
            if i == 1:
                inner_layer, layer_size = self.create_gated_layer(inner_layer, i + 2, layer_size, network_prefix,
                                                                  variable_list=variable_list)
            else:
                with tf.variable_scope(self.hidden_layer_name):
                    inner_layer, layer_size = self.create_layer(tf.nn.relu6, inner_layer, i + 2, layer_size,
                                                                self.network_size,
                                                                network_prefix, variable_list=variable_list)
        return inner_layer

    def create_last_layer(self, activation_function, inner_layer, network_size, num_actions,
                          network_prefix, last_layer_list=None):
        with tf.variable_scope(self.split_hidden_layer_name):
            inner_layer, layer_size = self.create_split_layers(tf.nn.relu6, inner_layer, self.network_size,
                                                               self.num_split_layers,
                                                               network_prefix, last_layer_list)
        return super().create_last_layer(activation_function, inner_layer, layer_size,
                                         num_actions, network_prefix, last_layer_list)

    def create_split_layers(self, activation_function, inner_layer, network_size,
                            num_split_layers, network_prefix, variable_list=None):
        split_layers = []
        num_actions = len(self.action_handler.get_split_sizes())

        for i in range(num_split_layers):
            for j, item in enumerate(self.action_handler.get_split_sizes()):
                name = str(self.action_handler.action_list_names[j]) + str(i)
                split_layers.append(self.create_layer(activation_function, inner_layer, 'split' + name,
                                                      network_size, network_size / num_actions, network_prefix + name,
                                                      variable_list=variable_list[j])[0])
            return split_layers, network_size / num_actions

    def create_savers(self):
        super().create_savers()
        self._create_layer_saver('actor_network', self.split_hidden_layer_name)
        self._create_layer_saver('actor_network', self.gated_layer_name)
