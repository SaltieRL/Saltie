from models import base_reinforcement
from models import base_model
import numpy as np
import tensorflow as tf
import random
import livedata.live_data_util as live_data_util


class BaseActorCritic(base_reinforcement.BaseReinforcement):
    frames_since_last_random_action = 0
    network_size = 128
    num_layers = 3
    last_row_variables = None
    actor_last_row_layer = None
    forced_frame_action = 500
    is_graphing = False
    keep_prob = 0.5
    reg_param = 0.001
    first_layer_name = 'first_layer'
    hidden_layer_name = 'hidden_layer'
    last_layer_name = 'last_layer'

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)
        if player_index >= 0:
            self.rotating_expected_reward_buffer = live_data_util.RotatingBuffer(player_index)

    def load_config_file(self):
        super().load_config_file()
        try:
            self.num_layers = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                             'num_layers')
        except:
            print('unable to load num_layers')

        try:
            self.network_size = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                      'num_width')
        except:
            print('unable to load the width of each layer')


        try:
            self.forced_frame_action = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                         'exploration_factor')
        except:
            print('unable to load exploration_factor')

    def get_input(self, model_input=None):
        if model_input is None:
            return super().get_input(self.input)
        else:
            return super().get_input(model_input)

    def _create_model(self, model_input):
        model_input = tf.check_numerics(model_input, 'model inputs')
        all_variable_list = []
        last_layer_list = [[] for _ in range(len(self.action_handler.get_split_sizes()))]
        with tf.name_scope("predict_actions"):
            # initialize actor-critic network
            with tf.variable_scope("actor_network", reuse=tf.AUTO_REUSE):
                self.policy_outputs = self.actor_network(model_input, variable_list=all_variable_list,
                                                         last_layer_list=last_layer_list)
            with tf.variable_scope("critic_network", reuse=tf.AUTO_REUSE):
                self.value_outputs = tf.reduce_mean(self.critic_network(model_input), name="Value_estimation")

            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            self.predicted_actions = \
                self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                              lambda split_tensor: tf.multinomial(split_tensor, 1))

        self.all_but_last_actor_layer = all_variable_list
        # get variable list
        self.actor_network_variables = all_variable_list + last_layer_list
        self.last_row_variables = last_layer_list
        self.critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

        self.split_action_scores = self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                                     lambda input_tensor: tf.identity(input_tensor),
                                                                     return_as_list=True)

        self.softmax = self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                                     lambda input_tensor: tf.nn.softmax(input_tensor),
                                                                     return_as_list=True)
        self.argmax = self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                                     lambda input_tensor: tf.argmax(tf.nn.softmax(input_tensor), axis=1),
                                                                     return_as_list=True)
        return self.predicted_actions, self.action_scores

    def create_reinforcement_training_model(self, model_input=None):
        converted_input = self.get_input(model_input)
        if self.batch_size > self.mini_batch_size:
            ds = tf.data.Dataset.from_tensor_slices((converted_input, self.taken_actions)).batch(self.mini_batch_size)
            self.iterator = ds.make_initializable_iterator()
            batched_input, batched_taken_actions = self.iterator.get_next()
        else:
            batched_input = converted_input
            batched_taken_actions = self.taken_actions
        with tf.name_scope("training_network"):
            self.discounted_rewards = self.discount_rewards(self.input_rewards, batched_input)
            with tf.variable_scope("actor_network", reuse=True):
                self.logprobs = self.actor_network(batched_input)

            with tf.variable_scope("critic_network", reuse=True):
                self.estimated_values = self.critic_network(batched_input)

            taken_actions = self.parse_actions(batched_taken_actions)

        self.train_op = self.create_training_op(self.logprobs, taken_actions)

    def create_training_op(self, logprobs, taken_actions):
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs,
                                                                            labels=taken_actions)
        return self.optimizer.minimize(cross_entropy_loss)

    def sample_action(self, input_state):
        # TODO: use this code piece when tf.multinomial gets better

        # epsilon-greedy exploration strategy
        if not self.is_evaluating and (random.random() * (self.forced_frame_action - self.frames_since_last_random_action)) < self.exploration:
            # print('random action used', str(self.frames_since_last_random_action))
            self.frames_since_last_random_action = 0
            return self.action_handler.get_random_option()
        else:
            self.frames_since_last_random_action += 1
            if self.is_graphing:
                estimated_reward, action_scores = self.sess.run([self.value_outputs, self.softmax],
                                                                {self.input_placeholder: input_state})
                # Average is bad metric but max is always 1 right now so using a more interesting graph
                self.rotating_expected_reward_buffer += estimated_reward
            else:
                action_scores = self.sess.run([self.softmax],
                                              {self.input_placeholder: input_state})[0]

            action = self.action_handler.optionally_split_numpy_arrays(action_scores,
                                              lambda input_array: np.argmax(np.random.multinomial(1, input_array[0])),
                                              is_already_split=True)

            return action

    def create_layer(self, activation_function, input, layer_number, input_size, output_size, network_prefix,
                     variable_list=None, dropout=True):
        weight_name = network_prefix + "W" + str(layer_number)
        bias_name = network_prefix + "b" + str(layer_number)
        W = tf.get_variable(weight_name, [input_size, output_size],
                             initializer=tf.random_normal_initializer())
        b = tf.get_variable(bias_name, [output_size],
                             initializer=tf.constant_initializer(0.0))
        layer_output = activation_function(tf.matmul(input, W) + b)
        if variable_list is not None:
            variable_list.append(W)
            variable_list.append(b)
        if self.is_training and dropout:
            layer_output = tf.nn.dropout(layer_output, self.keep_prob)
        self.stored_variables[weight_name] = W
        self.stored_variables[bias_name] = b
        return layer_output, output_size

    def actor_network(self, input_states, variable_list=None, last_layer_list=None):
        if last_layer_list is None:
            last_layer_list = [[] for _ in range(len(self.action_handler.get_split_sizes()))]
        # define policy neural network
        actor_prefix = 'actor'
        with tf.variable_scope(self.first_layer_name):
            layer1, _ = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, self.network_size, actor_prefix,
                                       variable_list=variable_list, dropout=False)

        inner_layer = self.create_hidden_layers(tf.nn.relu6, layer1, self.network_size, actor_prefix,
                                                variable_list=variable_list)

        output_layer = self.create_last_layer(tf.nn.sigmoid, inner_layer, self.network_size,
                                              self.num_actions, actor_prefix, last_layer_list=last_layer_list)
        return output_layer

    def critic_network(self, input_states):
        # define policy neural network
        critic_prefix = 'critic'
        critic_size = self.network_size
        # four sets of actions why not! :)
        critic_layers = self.num_layers
        output_size = 1
        with tf.variable_scope(self.first_layer_name):
            layer1, _ = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, critic_size, critic_prefix,
                                          dropout=False)
        with tf.variable_scope(self.hidden_layer_name):
            inner_layer = layer1
            for i in range(0, critic_layers - 2):
                inner_layer, _ = self.create_layer(tf.nn.relu6, inner_layer, i + 2, critic_size,
                                                   critic_size, critic_prefix)
        with tf.variable_scope(self.last_layer_name):
            output_layer, _ = self.create_layer(tf.nn.sigmoid, inner_layer, self.last_layer_name,
                                                critic_size, output_size, critic_prefix)
        return output_layer * 2.0 - 1.0

    def get_model_name(self):
        return 'base_actor_critic-' + str(self.num_layers) + '-layers'

    def parse_actions(self, taken_actions):
        return taken_actions

    def get_regularization_loss(self, variables, prefix=None):
        normalized_variables = [tf.reduce_sum(tf.nn.l2_loss(x) * self.reg_param)
                                for x in variables]

        reg_loss = tf.reduce_sum(normalized_variables, name=(prefix + '_reg_loss'))
        tf.summary.scalar(prefix + '_reg_loss', reg_loss)
        return reg_loss

    def create_hidden_layers(self, activation_function, input_layer, network_size, network_prefix,
                             variable_list=None):
        with tf.variable_scope(self.hidden_layer_name):
            inner_layer = input_layer
            for i in range(0, self.num_layers - 2):
                inner_layer, _ = self.create_layer(activation_function, inner_layer, i + 2, network_size,
                                                   network_size, network_prefix, variable_list=variable_list)
        return inner_layer

    def create_last_layer(self, activation_function, inner_layer, network_size, num_actions,
                          network_prefix, last_layer_list=None):
        with tf.variable_scope(self.last_layer_name):
            last_layer_name = 'last'
            if not self.action_handler.is_split_mode():
                self.actor_last_row_layer, _ = self.create_layer(activation_function, inner_layer, last_layer_name,
                                                                 network_size, num_actions, network_prefix,
                                                                 variable_list=last_layer_list, dropout=False)
                return self.actor_last_row_layer

            self.actor_last_row_layer = []
            for i, item in enumerate(self.action_handler.get_split_sizes()):
                self.actor_last_row_layer.append(self.create_layer(activation_function, inner_layer[i], last_layer_name,
                                                                   network_size, item, network_prefix + str(i),
                                                                   variable_list=last_layer_list[i], dropout=False)[0])

            return tf.concat(self.actor_last_row_layer, 1)

    def create_savers(self):
        self._create_network_saver('actor_network')
        self._create_network_saver('critic_network')

    def _create_network_saver(self, network_name):
        self._create_layer_saver(network_name, self.last_layer_name)
        self._create_layer_saver(network_name, self.first_layer_name)

        hidden_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=network_name + '/' + self.hidden_layer_name)
        reshaped_list = np.reshape(np.array(hidden_layers), [int(len(hidden_layers) / 2), 2])
        for i in range(len(reshaped_list)):
            layer_name = self.hidden_layer_name + str(i)
            saver_name = network_name + '_' + layer_name
            self.add_saver(saver_name, reshaped_list[i].tolist())

    def _create_layer_saver(self, network_name, layer_name):
        saver_name = network_name + '_' + layer_name + '_' + str(self.network_size)
        scope_name = network_name + "/" + layer_name
        self.add_saver(saver_name,
                       tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name))
