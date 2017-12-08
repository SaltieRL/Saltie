import collections
from models import base_reinforcement
from models import base_model
import numpy as np
import tensorflow as tf
import random


class BaseActorCritic(base_reinforcement.BaseReinforcement):
    frames_since_last_random_action = 0
    network_size = 128
    num_layers = 3
    forced_frame_action = 500

    def __init__(self, session,
                 state_dim,
                 num_actions,
                 action_handler,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 discount_factor=0.99,  # discount future rewards
                 ):
        super().__init__(session, state_dim, num_actions, action_handler, is_training,
                         optimizer, summary_writer, summary_every, discount_factor)

    def load_config_file(self):
        super().load_config_file()
        try:
            self.num_layers = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                             'num_layers')
        except:
            print('unable to load num_layers')

        try:
            self.forced_frame_action = self.config_file.getint(base_model.MODEL_CONFIGURATION_HEADER,
                                                         'exploration_factor')
        except:
            print('unable to load exploration_factor')

    def create_model(self, input):
        with tf.name_scope("predict_actions"):
            # initialize actor-critic network
            with tf.variable_scope("actor_network"):
                self.policy_outputs = self.actor_network(self.input)
            with tf.variable_scope("critic_network"):
                self.value_outputs = self.critic_network(self.input)

            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            self.predicted_actions = \
                self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                              lambda split_tensor: tf.multinomial(split_tensor, 1))

        # get variable list
        self.actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
        self.critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

        self.softmax = self.action_handler.run_func_on_split_tensors(self.action_scores,
                                                                     lambda input_tensor: tf.nn.softmax(input_tensor),
                                                                     return_as_list=True)
        return self.predicted_actions, self.action_scores

    def create_reinforcement_training_model(self):
        with tf.name_scope("training_network"):
            with tf.variable_scope("actor_network", reuse=True):
                self.logprobs = self.actor_network(self.input)

            with tf.variable_scope("critic_network", reuse=True):
                self.estimated_values = self.critic_network(self.input)

            self.train_op = self.action_handler.run_func_on_split_tensors([self.logprobs,
                                                          self.estimated_values,
                                                          self.taken_actions,
                                                          self.actor_network_variables,
                                                          self.critic_network_variables],
                                                                          self._create_training_op,
                                                                          return_as_list=True)

    def _create_training_op(self, logprobs, estimated_values, taken_actions, actor_network_variables, critic_network_variables):
        if len(taken_actions.get_shape()) == 2:
            taken_actions = tf.squeeze(taken_actions)

        # calculates the entropy loss from getting the label wrong
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs,
                                                                                 labels=taken_actions)
        # makes sure that everything is a list
        if not isinstance(actor_network_variables, collections.Sequence):
            actor_network_variables = [actor_network_variables]

        if not isinstance(critic_network_variables, collections.Sequence):
            critic_network_variables = [critic_network_variables]

        return self.create_training_op(cross_entropy_loss, estimated_values, actor_network_variables, critic_network_variables)

    def create_training_op(self, cross_entropy_loss, estimated_values, actor_network_variables, critic_network_variables):
        return self.optimizer.minimize(cross_entropy_loss)

    def sample_action(self, input_state):
        # TODO: use this code piece when tf.multinomial gets better
        # sample action from current policy
        # actions = self.session.run(self.predicted_actions, {self.input: input})[0]
        # return actions[0]

        # epsilon-greedy exploration strategy
        if not self.is_evaluating and (random.random() * (self.forced_frame_action - self.frames_since_last_random_action)) < self.exploration:
            # print('random action used', str(self.frames_since_last_random_action))
            self.frames_since_last_random_action = 0
            return self.action_handler.get_random_option()
        else:
            self.frames_since_last_random_action += 1

            action_scores = self.sess.run(self.softmax, {self.input: input_state})

            action = self.action_handler.\
                optionally_split_numpy_arrays(action_scores,
                                              lambda input_array: np.argmax(np.random.multinomial(1, input_array[0])),
                                              is_already_split=True)

            return action

    def _create_variables(self):
        super()._create_variables()

    def create_layer(self, activation_function, input, layer_number, input_size, output_size):
        W = tf.get_variable("W" + str(layer_number), [input_size, output_size],
                             initializer=tf.random_normal_initializer())
        b = tf.get_variable("b" + str(layer_number), [output_size],
                             initializer=tf.constant_initializer(0))
        h = activation_function(tf.matmul(input, W) + b)
        return h

    def actor_network(self, input_states):
        # define policy neural network
        layer1 = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, self.network_size)
        inner_layer = layer1
        print('num layers', self.num_layers)
        for i in range(0, self.num_layers - 2):
            inner_layer = self.create_layer(tf.nn.relu6, inner_layer, i + 2, self.network_size, self.network_size)
        output_layer = self.create_layer(tf.nn.sigmoid, inner_layer, self.num_layers, self.network_size, self.num_actions)

        return output_layer

    def critic_network(self, input_states):
        # define policy neural network
        layer1 = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, self.network_size)
        inner_layer = layer1
        for i in range(0, self.num_layers - 2):
            inner_layer = self.create_layer(tf.nn.relu6, inner_layer, i + 2, self.network_size, self.network_size)
        output_layer = self.create_layer(tf.nn.sigmoid, inner_layer, self.num_layers, self.network_size, self.num_actions)
        return output_layer

    def get_model_name(self):
        return 'base_actor_critic-' + str(self.num_layers) + '-layers'
