import collections
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
        if self.batch_size > self.mini_batch_size:
            ds = tf.data.Dataset.from_tensor_slices((self.input, self.taken_actions)).batch(self.mini_batch_size)
            self.iterator = ds.make_initializable_iterator()
            batched_input, batched_taken_actions = self.iterator.get_next()
        else:
            batched_input = self.input
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

            estimated_reward, action_scores = self.sess.run([self.estimated_values, self.softmax],
                                                            {self.input_placeholder: input_state})
            # Average is bad metric but max is always 1 right now so using a more interesting graph
            self.rotating_expected_reward_buffer += np.average(estimated_reward)


            action = self.action_handler.\
                optionally_split_numpy_arrays(action_scores,
                                              lambda input_array: np.argmax(np.random.multinomial(1, input_array[0])),
                                              is_already_split=True)

            return action

    def create_layer(self, activation_function, input, layer_number, input_size, output_size, network_prefix):
        weight_name = network_prefix + "W" + str(layer_number)
        bias_name = network_prefix + "b" + str(layer_number)
        W = tf.get_variable(weight_name, [input_size, output_size],
                             initializer=tf.random_normal_initializer())
        b = tf.get_variable(bias_name, [output_size],
                             initializer=tf.constant_initializer(0.0))
        h = activation_function(tf.matmul(input, W) + b)
        self.stored_variables[weight_name] = W
        self.stored_variables[bias_name] = b
        return h

    def actor_network(self, input_states):
        # define policy neural network
        actor_prefix = 'actor'
        layer1 = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, self.network_size, actor_prefix)
        inner_layer = layer1
        print('num layers', self.num_layers)
        for i in range(0, self.num_layers - 2):
            inner_layer = self.create_layer(tf.nn.relu6, inner_layer, i + 2, self.network_size,
                                            self.network_size, actor_prefix)
        with tf.variable_scope("last_layer"):
            output_layer = self.create_last_layer(tf.nn.sigmoid, inner_layer, self.num_layers,self.network_size,
                                                  self.num_actions, actor_prefix)

        return output_layer

    def critic_network(self, input_states):
        # define policy neural network
        critic_prefix = 'critic'
        critic_size = self.network_size
        # four sets of actions why not! :)
        output_size = self.num_actions #4
        layer1 = self.create_layer(tf.nn.relu6, input_states, 1, self.state_dim, critic_size, critic_prefix)
        inner_layer = layer1
        for i in range(0, self.num_layers - 2):
            inner_layer = self.create_layer(tf.nn.relu6, inner_layer, i + 2, critic_size,
                                            critic_size, critic_prefix)
        output_layer = self.create_layer(tf.nn.sigmoid, inner_layer, self.num_layers,
                                         critic_size, output_size, critic_prefix)
        return output_layer

    def get_model_name(self):
        return 'base_actor_critic-' + str(self.num_layers) + '-layers'

    def parse_actions(self, taken_actions):
        return taken_actions

    def create_last_layer(self, activation_function, inner_layer, num_layers, network_size, num_actions, actor_prefix):
        last_layer_name = 'last'
        if not self.action_handler.is_split_mode():
            self.actor_last_row_layer = (self.create_layer(activation_function, inner_layer, last_layer_name,
                                                           network_size, num_actions, actor_prefix))
            return self.actor_last_row_layer


        self.actor_last_row_layer = []
        for i, item in enumerate(self.action_handler.get_split_sizes()):
            self.actor_last_row_layer.append(self.create_layer(activation_function, inner_layer, last_layer_name, network_size, item, actor_prefix + str(i)))

        last_row_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network/last_layer")
        reshaped_list = np.reshape(np.array(last_row_variables), [int(len(last_row_variables) / 2), 2])
        self.last_row_variables = reshaped_list.tolist()
        return tf.concat(self.actor_last_row_layer, 1)
