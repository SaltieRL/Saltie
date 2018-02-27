# Defined as a generic bot, can use multiple models
from bot_code.modelHelpers.actions import action_factory
from bot_code.modelHelpers import reward_manager
from bot_code.modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from bot_code.utils.dynamic_import import get_field, get_class
import bot_code.livedata.live_data_util as live_data_util

import numpy as np
import tensorflow as tf
import time


class Agent:
    model_class = None
    previous_reward = 0
    previous_action = None
    previous_score = 0
    previous_enemy_goals = 0
    previous_owngoals = 0
    is_online_training = False
    is_graphing = True
    control_scheme = None

    def __init__(self, name, team, index, bot_parameters=None):
        self.last_frame_time = None
        self.config_file = bot_parameters
        self.index = index
        self.load_config_file()
        self.reward_manager = reward_manager.RewardManager()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()
        self.actions_handler = action_factory.get_handler(control_scheme=self.control_scheme)
        self.num_actions = self.actions_handler.get_logit_size()
        print('num_actions', self.num_actions)
        self.model = self.get_model_class()(self.sess,
                                            self.num_actions,
                                            input_formatter_info=[team, index],
                                            player_index=self.index,
                                            action_handler=self.actions_handler,
                                            config_file=bot_parameters,
                                            is_training=False)

        self.model.add_summary_writer('random_packet', is_replay=True)

        self.model.batch_size = 1
        self.model.mini_batch_size = 1

        self.model.is_graphing = self.is_graphing

        self.model.is_online_training = self.is_online_training

        self.model.apply_feature_creation(TensorflowFeatureCreator())

        try:
            self.model.create_model(self.model.get_input_placeholder())
        except TypeError as e:
            raise Exception('failed to create model') from e

        if self.model.is_training and self.model.is_online_training:
            self.model.create_reinforcement_training_model()

        self.model.create_savers()

        self.model.initialize_model()
        if self.is_graphing:
            self.rotating_real_reward_buffer = live_data_util.RotatingBuffer(self.index + 10)

    def load_config_file(self):
        if self.config_file is None:
            return
        # read file code here

        model_package = self.config_file.get('model_package')
        model_name = self.config_file.get('model_name')

        try:
            self.is_graphing = self.config_file.getboolean('should_graph', self.is_graphing)
        except:
            print('not generating graph data')

        try:
            self.is_online_training = self.config_file.getboolean('train_online', self.is_online_training)
        except:
            print('not training online')
        try:
            control_scheme = self.config_file.get('control_scheme', 'default_scheme')
        except Exception as e:
            control_scheme = 'default_scheme'

        print('getting model from', model_package)
        print('name of model', model_name)
        self.model_class = get_class(model_package, model_name)
        self.control_scheme = get_field('modelHelpers.actions.action_factory', control_scheme)

    def get_model_class(self):
        if self.model_class is None:
            print('Invalid model using default')
            return None
        else:
            return self.model_class

    def get_reward(self, input_state):
        reward = self.reward_manager.get_reward(input_state)
        return reward[0] + reward[1]

    def get_output_vector(self, game_tick_packet):
        frame_time = 0.0
        if self.last_frame_time is not None:
            frame_time = game_tick_packet.gameInfo.TimeSeconds - self.last_frame_time
        self.last_frame_time = game_tick_packet.gameInfo.TimeSeconds
        input_state = self.model.create_input_array(game_tick_packet, frame_time)
        if self.model.state_dim != len(input_state):
            print('wrong input size', self.index, len(input_state))
            return self.actions_handler.create_controller_from_selection(
                self.actions_handler.get_random_option())  # do not return anything

        if self.model.is_training and self.is_online_training:
            if self.previous_action is not None:
                self.model.store_rollout(input_state, self.previous_action, 0)
        if self.is_graphing:
            reward = self.get_reward(input_state)
            self.rotating_real_reward_buffer += reward

        reshaped = np.array(input_state).reshape((1, -1))
        output = np.argwhere(np.isnan(reshaped))
        if len(output) > 0:
            print('nan indexes', output)
            for index in output:
                reshaped[index[0]][index[1]] = 0

        action = self.model.sample_action(reshaped)
        if action is None:
            print("invalid action no type returned")
            action = self.actions_handler.get_random_option()
        self.previous_action = action
        controller_selection = self.actions_handler.create_controller_from_selection(action)
        controller_selection = [max(-1, min(1, control)) for control in controller_selection]
        return controller_selection

    def create_model_hash(self):
        try:
            return self.model.create_model_hash()
        except Exception as e:
            print('creating hash exception', e)
            return 0
