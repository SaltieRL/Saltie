# Defined as a generic bot, can use multiple models
import random

import numpy as np
import tensorflow as tf

from conversions.input_formatter import InputFormatter
from modelHelpers import action_handler
from modelHelpers import reward_manager
from models.actor_critic import policy_gradient


class Agent:
    previous_reward = 0
    previous_action = None
    previous_score = 0
    previous_enemy_goals = 0
    previous_owngoals = 0
    def __init__(self, name, team, index):
        self.index = index
        self.inp = InputFormatter(team, index)
        self.reward_manager = reward_manager.RewardManager()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        #self.sess = tf.Session()
        writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))
        self.actions_handler = action_handler.ActionHandler(split_mode=True)
        self.state_dim = self.inp.get_state_dim_with_features()
        self.num_actions = self.actions_handler.get_action_size()
        print('num_actions', self.num_actions)
        self.model = self.get_model_class()(self.sess,
                                            self.state_dim,
                                            self.num_actions,
                                            action_handler=self.actions_handler,
                                            summary_writer=writer)
        self.model.initialize_model()

    def get_model_class(self):
        #return nnatba.NNAtba
        #return rnn_atba.RNNAtba
        return policy_gradient.PolicyGradient

    def get_reward(self, input_state):
        reward = self.reward_manager.get_reward(input_state)
        return reward

    def get_output_vector(self, game_tick_packet):
        input_state, features = self.inp.create_input_array(game_tick_packet)
        input_state = np.append(input_state, features)
        if self.state_dim != len(input_state):
            print('wrong input size', self.index, len(input_state))
            return self.actions_handler.create_controller_from_selection(
                self.actions_handler.get_random_option()) # do not return anything

        if self.model.is_training:
            reward = self.get_reward(input_state)

            if self.previous_action is not None:
                self.model.store_rollout(input_state, self.previous_action, reward)

        action = self.model.sample_action(np.array(input_state).reshape((1, -1)))
        if action is None:
            print("invalid action no type returned")
            action = self.actions_handler.get_random_option()
        self.previous_action = action
        return self.actions_handler.create_controller_from_selection(action)
