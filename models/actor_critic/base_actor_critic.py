from models import base_reinforcement
from models import base_model
import numpy as np
import tensorflow as tf
import random
import livedata.live_data_util as live_data_util
import collections


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
    layers = []

    # tensorflow objects
    discounted_rewards = None
    estimated_values = None
    logprobs = None

    def __init__(self, session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None
                 ):
        super().__init__(session, num_actions,
                         input_formatter_info=input_formatter_info,
                         player_index=player_index,
                         action_handler=action_handler,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)
        if player_index >= 0:
            self.rotating_expected_reward_buffer = live_data_util.RotatingBuffer(player_index)

    def printParameters(self):
        super().printParameters()
        print('Actor Critic Parameters')
        print('network size', self.network_size)
        print('number of layers', self.num_layers)
        print('keep probability', self.keep_prob)
        print('regulation parameter', self.reg_param)

    def get_activation(self):
        return tf.nn.elu  # tf.nn.relu6

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

        try:
            self.keep_prob = self.config_file.getfloat(base_model.MODEL_CONFIGURATION_HEADER,
                                                     'keep_probability')
        except:
            print('unable to load keep_probability')

    def smart_argmax(self, input_tensor, index):
        if not self.action_handler.is_classification(index):
            # input_tensor = tf.Print(input_tensor, [input_tensor], str(index))
            return tf.squeeze(input_tensor, axis=1)
        argmax_index = tf.cast(tf.argmax(input_tensor, axis=1), tf.int32)
        indexer = tf.range(0, self.batch_size)
        slicer_data = tf.stack([indexer, argmax_index], axis=1)
        sliced_tensor = tf.gather_nd(input_tensor, slicer_data)
        condition = tf.greater(sliced_tensor, self.action_threshold)
        true = tf.cast(condition, tf.int32)
        false = 1 - tf.cast(condition, tf.int32)

        random_action = tf.squeeze(tf.multinomial(tf.nn.softmax(input_tensor), 1))

        return argmax_index * true + false * tf.cast(random_action, tf.int32)

    def _create_model(self, model_input):
        model_input = tf.check_numerics(model_input, 'model inputs')
        all_variable_list = []
        last_layer_list = [[] for _ in range(len(self.action_handler.get_action_sizes()))]
        with tf.name_scope("predict_actions"):
            # initialize actor-critic network
            with tf.variable_scope("actor_network", reuse=tf.AUTO_REUSE):
                self.policy_outputs = self.actor_network(model_input, variable_list=all_variable_list,
                                                         last_layer_list=last_layer_list,
                                                         layers_list=self.layers)
            with tf.variable_scope("critic_network", reuse=tf.AUTO_REUSE):
                self.value_outputs = tf.reduce_mean(self.critic_network(model_input), name="Value_estimation")

            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            self.predicted_actions = \
                self.action_handler.run_func_on_split_tensors(tf.nn.softmax(self.policy_outputs),
                                                              lambda split_tensor: tf.multinomial(split_tensor, 1))

        self.all_but_last_actor_layer = all_variable_list
        # get variable list
        self.actor_network_variables = all_variable_list + last_layer_list
        self.last_row_variables = last_layer_list
        self.critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

        self.split_action_scores = self.action_handler.run_func_on_split_tensors(self.policy_outputs,
                                                                     lambda input_tensor: tf.identity(input_tensor),
                                                                     return_as_list=True)

        self.softmax = self.action_handler.run_func_on_split_tensors(self.policy_outputs,
                                                                     lambda input_tensor: tf.nn.softmax(input_tensor),
                                                                     return_as_list=True)
        self.argmax = self.action_handler.run_func_on_split_tensors(self.policy_outputs,
                                                                     lambda input_tensor: tf.argmax(
                                                                         tf.nn.softmax(input_tensor), axis=1),
                                                                     return_as_list=True)
        indexes = np.arange(0, self.action_handler.get_number_actions(), 1).tolist()
        self.smart_max = self.action_handler.run_func_on_split_tensors([self.policy_outputs, indexes],
                                                                       self.smart_argmax,
                                                                       return_as_list=True)
        return self.predicted_actions, self.action_scores

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        converted_input = self.get_input(model_input)

        if taken_actions is None:
            actions_input = self.get_labels_placeholder()
        else:
            actions_input = taken_actions

        batched_input, batched_taken_actions = self.create_batched_inputs([converted_input, actions_input])

        with tf.name_scope("training_network"):
            self.discounted_rewards = tf.constant(0.0)
            with tf.variable_scope("actor_network", reuse=True):
                self.logprobs = self.actor_network(batched_input)

            with tf.variable_scope("critic_network", reuse=True):
                self.estimated_values = tf.constant(1.0)

            taken_actions = self.parse_actions(batched_taken_actions)

        self.log_output_data()

        self.train_op = self.create_training_op(self.logprobs, taken_actions)

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
        # TODO: use tf.multinomial when it gets better

        # epsilon-greedy exploration strategy
        if not self.is_evaluating and (random.random() * (self.forced_frame_action -
                                                          self.frames_since_last_random_action)) < self.exploration:
            self.frames_since_last_random_action = 0
            return self.action_handler.get_random_option()
        else:
            self.frames_since_last_random_action += 1
            if self.is_graphing:
                estimated_reward, action_scores = self.sess.run([self.value_outputs, self.smart_max],
                                                                {self.input_placeholder: input_state})
                # Average is bad metric but max is always 1 right now so using a more interesting graph
                self.rotating_expected_reward_buffer += estimated_reward
            else:
                action_scores = self.sess.run([self.smart_max],
                                              {self.input_placeholder: input_state})
                print(action_scores)

            action_scores = np.array(action_scores).flatten()
            return action_scores

    def create_layer(self, activation_function, input, layer_number, input_size, output_size, network_prefix,
                     variable_list=None, dropout=True):
        weight_name = network_prefix + "W" + str(layer_number)
        bias_name = network_prefix + "b" + str(layer_number)
        W = tf.get_variable(weight_name, [input_size, output_size],
                             initializer=tf.random_normal_initializer())
        b = tf.get_variable(bias_name, [output_size],
                             initializer=tf.random_normal_initializer())
        if activation_function is not None:
            layer_output = activation_function(tf.matmul(input, W) + b)
        else:
            layer_output = tf.matmul(input, W) + b
        if variable_list is not None:
            variable_list.append(W)
            variable_list.append(b)
        if self.is_training and dropout:
            layer_output = tf.nn.dropout(layer_output, self.keep_prob)
        self.stored_variables[weight_name] = W
        self.stored_variables[bias_name] = b
        return layer_output, output_size

    def actor_network(self, input_states, variable_list=None, last_layer_list=None, layers_list=[]):
        if last_layer_list is None:
            last_layer_list = [[] for _ in range(len(self.action_handler.get_action_sizes()))]
        # define policy neural network
        actor_prefix = 'actor'
        activation = self.get_activation()
        # input_states = tf.Print(input_states, [input_states], summarize=self.network_size, message='')
        with tf.variable_scope(self.first_layer_name):
            layer1, _ = self.create_layer(activation, input_states, 1, self.state_feature_dim, self.network_size, actor_prefix,
                                       variable_list=variable_list, dropout=False)
        layers_list.append([layer1])

        # layer1 = tf.Print(layer1, [layer1], summarize=self.network_size, message='')

        inner_layer, output_size = self.create_hidden_layers(activation, layer1, self.network_size, actor_prefix,
                                                variable_list=variable_list, layers_list=layers_list)

        output_layer = self.create_last_layer(tf.nn.sigmoid, inner_layer, output_size,
                                              self.num_actions, actor_prefix,
                                              last_layer_list=last_layer_list, layers_list=layers_list)

        return output_layer

    def critic_network(self, input_states):
        # define policy neural network
        critic_prefix = 'critic'
        critic_size = self.network_size
        # four sets of actions why not! :)
        critic_layers = self.num_layers
        output_size = 1
        with tf.variable_scope(self.first_layer_name):
            layer1, _ = self.create_layer(tf.nn.relu6, input_states, 1, self.state_feature_dim, critic_size, critic_prefix,
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

    def log_output_data(self):
        """Logs the output of the last layer of the model"""
        with tf.name_scope('model_output'):
            for i in range(self.action_handler.get_number_actions()):
                variable_name = str(self.action_handler.action_list_names[i])
                tf.summary.histogram(variable_name + '_output', self.actor_last_row_layer[i])

    def get_regularization_loss(self, variables, prefix=None):
        normalized_variables = [tf.reduce_sum(tf.nn.l2_loss(x) * self.reg_param)
                                for x in variables]

        reg_loss = tf.reduce_sum(normalized_variables, name=(prefix + '_reg_loss'))
        tf.summary.scalar(prefix + '_reg_loss', reg_loss)
        return tf.constant(0.0)  # reg_loss

    def create_hidden_layers(self, activation_function, input_layer, network_size, network_prefix, variable_list=None,
                             layers_list=[]):
        with tf.variable_scope(self.hidden_layer_name):
            inner_layer = input_layer
            for i in range(0, self.num_layers - 2):
                inner_layer, _ = self.create_layer(activation_function, inner_layer, i + 2, network_size,
                                                   network_size, network_prefix, variable_list=variable_list)
                layers_list.append(inner_layer)
        return inner_layer, network_size

    def create_last_layer(self, activation_function, inner_layer, network_size, num_actions, network_prefix,
                          last_layer_list=None, layers_list=[]):
        with tf.variable_scope(self.last_layer_name):
            last_layer_name = 'final'
            if not self.action_handler.is_split_mode():
                self.actor_last_row_layer, _ = self.create_layer(activation_function, inner_layer[0], last_layer_name,
                                                                 network_size, num_actions, network_prefix,
                                                                 variable_list=last_layer_list[0], dropout=False)

                return self.actor_last_row_layer

            self.actor_last_row_layer = []
            if not isinstance(inner_layer, collections.Sequence):
                inner_layer = [inner_layer] * self.action_handler.get_number_actions()
            for i, item in enumerate(self.action_handler.get_action_sizes()):
                variable_name = str(self.action_handler.action_list_names[i])
                with tf.variable_scope(variable_name):
                    fixed_activation = self.action_handler.get_last_layer_activation_function(activation_function, i)
                    layer = self.create_layer(fixed_activation, inner_layer[i], last_layer_name,
                                                                       network_size, item, network_prefix,
                                                                       variable_list=last_layer_list[i], dropout=False)[0]
                    scaled_layer = self.action_handler.scale_layer(layer, i)
                    self.actor_last_row_layer.append(scaled_layer)
            layers_list.append(self.actor_last_row_layer)
            return tf.concat(self.actor_last_row_layer, 1)

    def create_savers(self):
        super().create_savers()
        self._create_network_saver('actor_network')
        self._create_network_saver('critic_network')

    def _create_network_saver(self, network_name):
        self._create_layer_saver(network_name, self.first_layer_name, self.state_feature_dim)
        self._create_hidden_row_saver(network_name)
        self._create_last_row_saver(network_name)

    def _create_last_row_saver(self, network_name):
        split_las_layer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=network_name + '/' + self.last_layer_name)
        reshaped_list = np.reshape(np.array(split_las_layer), [int(len(split_las_layer) / 2), 2])
        if len(reshaped_list) == 1:
            self._create_layer_saver(network_name, self.last_layer_name,
                                     variable_list=reshaped_list[0].tolist())
            return
        for i in range(len(reshaped_list)):
            self._create_layer_saver(network_name, self.last_layer_name,
                                     extra_info=self.action_handler.action_list_names[i],
                                     variable_list=reshaped_list[i].tolist())

    def _create_hidden_row_saver(self, network_name):
        hidden_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope=network_name + '/' + self.hidden_layer_name)
        reshaped_list = np.reshape(np.array(hidden_layers), [int(len(hidden_layers) / 2), 2])
        for i in range(len(reshaped_list)):
            self._create_layer_saver(network_name, self.hidden_layer_name,
                                     extra_info=str(i),
                                     variable_list=reshaped_list[i].tolist())

    def _create_layer_saver(self, network_name, layer_name, extra_info=None, variable_list=None):
        details = ''
        if extra_info is not None:
            details = '_' + str(extra_info).replace(' ', '').replace('[', '')\
                .replace(']', '').replace(',', '_').replace('\'','')

        saver_name = network_name + '_' + layer_name + '_' + str(self.network_size) + details
        scope_name = network_name + "/" + layer_name
        if variable_list is None:
            self.add_saver(saver_name,
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name))
        else:
            self.add_saver(saver_name, variable_list)

    def get_variables_activations(self):
        unified_layers = np.array(self.all_but_last_actor_layer).reshape((-1, 2))
        split_layers = np.array(self.last_row_variables).reshape((-1, len(self.last_row_variables), 2))
        unified_layers = self.sess.run(unified_layers.tolist())
        split_layers = self.sess.run(split_layers.tolist())
        network_variables = []
        for element in unified_layers:
            layer = element + ['relu']
            network_variables.append([layer])
        for i, layer in enumerate(split_layers):
            split_layer = []
            for j, element in enumerate(layer):
                if i == len(split_layers) - 1:
                    output_type = ['sigmoid' if self.action_handler.is_classification(j) else 'none']
                else:
                    output_type = ['relu']
                layer = element + output_type
                split_layer.append(layer)
            network_variables.append(split_layer)
        return network_variables

    def get_activations(self, input_array=None):
        layer_activations = self.sess.run(self.layers, feed_dict={self.get_input_placeholder(): input_array})
        return layer_activations
