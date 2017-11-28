# Defined as a generic bot, can use multiple models
import numpy as np
import random
import tensorflow as tf

from conversions.input_formatter import InputFormatter
from modelHelpers import action_handler
from modelHelpers import reward_manager
from models import actor_critic_wrapper
from models import rnn_atba
from models import nnatba


class Agent:
    previous_reward = 0
    previous_action = 0
    previous_score = 0
    previous_enemy_goals = 0
    previous_owngoals = 0
    def __init__(self, name, team, index):
        self.index = index
        self.inp = InputFormatter(team, index)
        self.reward_manager = reward_manager.RewardManager(name, team, index, self.inp)
        #config = tf.ConfigProto(
        #    device_count={'GPU': 0}
        #)
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))
        self.actions_handler = action_handler.ActionHandler(split_mode=True)
        self.state_dim = 197
        self.num_actions = self.actions_handler.get_action_size()
        print('num_actions', self.num_actions)
        self.model = self.get_model_class()(self.sess,
                                            self.state_dim,
                                            self.num_actions,
                                            self.actions_handler,
                                            summary_writer=writer)
        self.model.initialize_model()

    def get_model_class(self):
        #return nnatba.NNAtba
        return rnn_atba.RNNAtba

    def get_reward(self, packet):
        reward = self.reward_manager.get_reward(packet)
        self.reward_manager.update_from_packet(packet)
        return reward

    def get_output_vector(self, game_tick_packet):
        state, features = self.inp.create_input_array(game_tick_packet)
        state = np.append(state, features)
        if self.state_dim != len(state):
            print('wrong input size', self.index, len(state))
            return self.actions_handler.create_controller_from_selection(
                self.actions_handler.get_random_option()) # do not return anything

        reward = self.get_reward(game_tick_packet)

        self.model.store_rollout(state, self.previous_action, reward)

        action = self.model.sample_action(np.array(state).reshape((1, -1)))
        if action is None:
            print("invalid action no type returned")
        if random.random() < 0.00005 or action is None:
            action = self.actions_handler.get_random_option()
        self.previous_action = action
        return self.actions_handler.create_controller_from_selection(action)
