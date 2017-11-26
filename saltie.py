# Defined as a generic bot, can use multiple models
import numpy as np
import random
import tensorflow as tf

from conversions.input_formatter import InputFormatter
from modelHelpers import option_handler
from modelHelpers import reward_manager
from models import actor_critic_wrapper
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
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))
        self.options = option_handler.createOptions()
        self.state_dim = 195
        self.num_actions = len(self.options)
        print('num_actions', self.num_actions)
        self.model = self.get_model_class()(self.sess,
                                            self.state_dim,
                                            self.num_actions,
                                            summary_writer=writer)

    def get_model_class(self):
        return nnatba.NNAtba

    def get_reward(self, packet):
        reward = self.reward_manager.get_reward(packet)
        self.reward_manager.update_from_packet(packet)
        return reward

    def get_output_vector(self, game_tick_packet):
        state = self.inp.create_input_array(game_tick_packet)
        if self.state_dim != len(state):
            print('wrong input size', self.index, len(state))
            return self.options[0] # do not return anything
        reward = self.get_reward(game_tick_packet)

        self.model.store_rollout(state, self.previous_action, reward)

        action = self.model.sample_action(np.array(state).reshape((1, -1)))
        print('selected action ' + str(action))
        if action is None:
            print("invalid action no type returned")
        if random.random() < 0.05 or action is None:
            action = random.randint(0, self.num_actions)
            if action == 256:
                print('f_in rand int', action)
        self.previous_action = action
        if action >= self.num_actions or action < 0:
            print (action, len(self.options))
            return self.options[0]
        return self.options[action]
