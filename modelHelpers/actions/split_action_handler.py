import numpy as np
import tensorflow as tf

from modelHelpers.actions.action_handler import ActionHandler, ActionMap


class SplitActionHandler(ActionHandler):
    actions_split = []
    split_action_sizes = []
    movement_actions = []
    combo_actions = []
    action_list_size = 0

    def __init__(self):
        super().__init__()

    def reset(self):
        self.actions_split = []
        self.split_action_sizes = []
        self.action_list_size = 0
        self.movement_actions = []
        self.combo_actions = []

    def create_actions(self):
        self.reset()

        """
        Creates all variations of all of the actions.

        controller options = [throttle, steer, pitch, steer, roll, jump, boost, handbrake]
        :return: A combination of all actions. This is an array of an array
        """

        steer = np.arange(-1, 1.5, .5)
        pitch = np.arange(-1, 1.5, .5)
        roll = np.arange(-1, 1.5, .5)
        throttle = np.arange(-1, 2, 1)
        jump = [False, True]
        boost = [False, True]
        handbrake = [False, True]
        action_list = [throttle, jump, boost, handbrake]
        self.action_list_size = len(action_list)
        # 24 + 5 + 5 + 5 = 39
        button_combo = list(np.itertools.product(*action_list))
        actions = []
        split_actions_sizes = []
        actions.append(steer)
        actions.append(pitch)
        actions.append(roll)
        self.movement_actions = tf.constant(np.array(actions), shape=[len(actions), len(steer)])
        self.combo_actions = tf.constant(button_combo)
        self.action_list_names = ['steer', 'pitch', 'yaw', 'combo']
        actions.append(button_combo)
        for i in actions:
            split_actions_sizes.append(len(i))
        self.actions_split = actions
        self.split_action_sizes = split_actions_sizes
        return actions

    def create_action_map(self):
        return ActionMap(self.actions[3])

    def is_split_mode(self):
        return True

    def get_action_sizes(self):
        return self.split_action_sizes

    def get_action_size(self):
        """
        :return: the size of the logits layer in a model
        """
        if not self.split_mode:
            return super().get_action_size()

        counter = 0
        for action in self.actions_split:
            counter += len(action)
        return counter

    def create_action_label(self, real_action):
        indexes = self._create_split_indexes(real_action)
        return self._create_split_label(indexes)

    def _create_split_label(self, action_indexes):
        encoding = np.zeros(39)
        encoding[action_indexes[0] + 0] = 1
        encoding[action_indexes[1] + 5] = 1
        encoding[action_indexes[2] + 10] = 1
        encoding[action_indexes[3] + 15] = 1
        return encoding

    def _create_split_indexes(self, real_action):
        steer = real_action[1]
        yaw = real_action[3]
        if steer != yaw and abs(steer) < abs(yaw):
            # only take the larger magnitude number
            steer = yaw

        steer_index = self._find_closet_real_number(steer)
        pitch_index = self._find_closet_real_number(real_action[2])
        roll_index = self._find_closet_real_number(real_action[4])
        button_combo = self.action_map.get_key([round(real_action[0]), real_action[5], real_action[6], real_action[7]])

        return [steer_index, pitch_index, roll_index, button_combo]
