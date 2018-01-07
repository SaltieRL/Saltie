# Defined as a generic bot, can use multiple models
from conversions.input import input_formatter
from conversions.input.input_formatter import InputFormatter
import importlib
import inspect
from modelHelpers.actions import action_handler, action_factory, dynamic_action_handler
from modelHelpers import reward_manager
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from models.actor_critic import policy_gradient
import livedata.live_data_util as live_data_util

import numpy as np
import random
import tensorflow as tf
import time

MODEL_CONFIGURATION_HEADER = 'Model Configuration'


class Agent:
    model_class = None
    previous_reward = 0
    previous_action = None
    previous_score = 0
    previous_enemy_goals = 0
    previous_owngoals = 0
    is_online_training = False
    is_graphing = True

    def __init__(self, name, team, index, config_file=None):
        self.last_frame_time = None
        self.config_file = config_file
        self.index = index
        self.load_config_file()
        self.inp = InputFormatter(team, index)
        self.reward_manager = reward_manager.RewardManager()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()
        writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))
        self.actions_handler = action_factory.get_handler(control_scheme=dynamic_action_handler.super_split_scheme)
        self.state_dim = input_formatter.get_state_dim()
        self.num_actions = self.actions_handler.get_action_size()
        print('num_actions', self.num_actions)
        self.model = self.get_model_class()(self.sess,
                                            self.state_dim,
                                            self.num_actions,
                                            player_index=self.index,
                                            action_handler=self.actions_handler,
                                            summary_writer=writer,
                                            is_training=True)

        self.model.is_graphing = self.is_graphing

        self.model.is_online_training = self.is_online_training

        self.model.apply_feature_creation(TensorflowFeatureCreator())

        self.model.create_model()

        if self.model.is_training and self.model.is_online_training:
            self.model.create_reinforcement_training_model()

        self.model.initialize_model()
        if self.is_graphing:
            self.rotating_real_reward_buffer = live_data_util.RotatingBuffer(self.index + 10)

    def load_config_file(self):
        if self.config_file is None:
            return
        # read file code here

        model_package = self.config_file.get(MODEL_CONFIGURATION_HEADER, 'model_package')
        model_name = self.config_file.get(MODEL_CONFIGURATION_HEADER, 'model_name')

        try:
            self.is_graphing = self.config_file.getboolean(MODEL_CONFIGURATION_HEADER, 'should_graph')
        except:
            print('not generating graph data')

        try:
            self.is_online_training = self.config_file.getboolean(MODEL_CONFIGURATION_HEADER, 'train_online')
        except:
            print('not training online')

        print('getting model from', model_package)
        print('name of model', model_name)
        model_package = importlib.import_module(model_package)
        module_classes = inspect.getmembers(model_package, inspect.isclass)
        for class_group in module_classes:
            if class_group[0] == model_name:
                self.model_class = class_group[1]
        if self.model_class is not None:
            self.model_class.config_file = self.config_file

    def get_model_class(self):
        if self.model_class is None:
            print('Invalid model using default')
            return policy_gradient.PolicyGradient
        else:
            return self.model_class

    def get_reward(self, input_state):
        reward = self.reward_manager.get_reward(input_state)
        return reward[0] + reward[1]

    def get_output_vector(self, game_tick_packet):
        frame_time = 0.0
        if self.last_frame_time is not None:
            frame_time = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        input_state = self.inp.create_input_array(game_tick_packet, frame_time)
        if self.state_dim != len(input_state):
            print('wrong input size', self.index, len(input_state))
            return self.actions_handler.create_controller_from_selection(
                self.actions_handler.get_random_option())  # do not return anything

        if self.model.is_training and self.is_online_training:
            if self.previous_action is not None:
                self.model.store_rollout(input_state, self.previous_action, 0)
        if self.is_graphing:
            reward = self.get_reward(input_state)
            self.rotating_real_reward_buffer += reward

        action = self.model.sample_action(np.array(input_state).reshape((1, -1)))
        if action is None:
            print("invalid action no type returned")
            action = self.actions_handler.get_random_option()
        self.previous_action = action
        return self.actions_handler.create_controller_from_selection(action)

    def create_model_hash(self):
        try:
            return self.model.create_model_hash()
        except Exception as e:
            print('creating hash exception', e)
            return 0
