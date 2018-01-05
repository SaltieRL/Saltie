from models.actor_critic.policy_gradient import PolicyGradient
import tensorflow as tf


class TutorialModel(PolicyGradient):
    num_split_layers = 1


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

    def calculate_loss_of_actor(self, cross_entropy_loss, wrongness, index):
        if self.action_handler.action_list_names[index] != 'combo':
            return cross_entropy_loss * wrongness, False
        return cross_entropy_loss, False

    def actor_network(self, input_states, variable_list=None, last_layer_list=None):
        if last_layer_list is None:
            last_layer_list = [[] for _ in range(len(self.action_handler.get_split_sizes()))]
        # define policy neural network
        actor_prefix = 'actor'
        with tf.variable_scope(self.first_layer_name):
            layer1 = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, self.network_size, actor_prefix,
                                       variable_list=variable_list, dropout=False)

        with tf.variable_scope(self.hidden_layer_name):
            inner_layer = layer1
            print('num layers', self.num_layers)
            for i in range(0, self.num_layers - 2 - self.num_split_layers):
                inner_layer = self.create_layer(tf.nn.relu6, inner_layer, i + 2, self.network_size,
                                                self.network_size, actor_prefix, variable_list=variable_list)
        with tf.variable_scope(self.split_hidden_layer_name):
            inner_layer, layer_size = self.create_split_layers(tf.nn.relu6, inner_layer, self.network_size,
                                                               self.num_split_layers,
                                                               actor_prefix, last_layer_list)

        with tf.variable_scope(self.last_layer_name):
            output_layer = self.create_last_layer(tf.nn.sigmoid, inner_layer, layer_size,
                                                  self.num_actions, actor_prefix, last_layer_list)

        return output_layer

    def create_split_layers(self, activation_function, inner_layer, network_size,
                            num_split_layers, actor_prefix, variable_list=None):
        split_layers = []
        num_actions = len(self.action_handler.get_split_sizes())

        for i in range(num_split_layers):
            for j, item in enumerate(self.action_handler.get_split_sizes()):
                name = str(self.action_handler.action_list_names[j])
                split_layers.append(self.create_layer(activation_function, inner_layer, 'split' + name,
                                                      network_size, network_size / num_actions, actor_prefix + name,
                                                      variable_list=variable_list[j]))
            return split_layers, network_size / num_actions
