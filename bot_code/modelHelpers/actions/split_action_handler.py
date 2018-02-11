import random
import itertools
import collections
import numpy as np
import tensorflow as tf

from bot_code.modelHelpers.actions.action_handler import ActionHandler, ActionMap


class SplitActionHandler(ActionHandler):
    actions = []
    action_sizes = []
    movement_actions = []
    tensorflow_combo_actions = []

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        self.actions = []
        self.action_sizes = []
        self.movement_actions = []
        self.tensorflow_combo_actions = []

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
        self.combo_list = action_list
        # 24 + 5 + 5 + 5 = 39
        button_combo = list(itertools.product(*action_list))
        actions = []
        split_actions_sizes = []
        actions.append(steer)
        actions.append(pitch)
        actions.append(roll)
        self.movement_actions = tf.constant(np.array(actions), shape=[len(actions), len(steer)])
        self.tensorflow_combo_actions = tf.constant(button_combo)
        self.action_list_names = ['steer', 'pitch', 'yaw', 'combo']
        actions.append(button_combo)
        for i in actions:
            split_actions_sizes.append(len(i))
        self.action_sizes = split_actions_sizes
        return actions

    def create_action_map(self):
        return ActionMap(self.actions[3])

    def is_split_mode(self):
        return True

    def get_number_actions(self):
        return len(self.actions)

    def get_action_sizes(self):
        return self.action_sizes

    def get_logit_size(self):
        """
        :return: the size of the logits layer in a model
        """

        counter = 0
        for action in self.actions:
            counter += len(action)
        return counter

    def _create_one_hot_encoding(self, action_indexes):
        encoding = np.zeros(39)
        encoding[action_indexes[0] + 0] = 1
        encoding[action_indexes[1] + 5] = 1
        encoding[action_indexes[2] + 10] = 1
        encoding[action_indexes[3] + 15] = 1
        return encoding

    def create_action_index(self, real_action):
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

    def create_controller_from_selection(self, action_selection):
        if len(action_selection) != len(self.actions):
            print('ACTION SELECTION IS NOT THE SAME LENGTH returning invalid action data')
            return [0, 0, 0, 0, 0, False, False, False]
        steer = self.actions[0][action_selection[0]]
        pitch = self.actions[1][action_selection[1]]
        roll = self.actions[2][action_selection[2]]
        button_combo = self.actions[3][action_selection[3]]
        throttle = button_combo[0]
        jump = button_combo[1]
        boost = button_combo[2]
        handbrake = button_combo[3]
        controller_option = [throttle, steer, pitch, steer, roll, jump, boost, handbrake]
        # print(controller_option)
        return controller_option

    def create_tensorflow_controller_from_selection(self, action_selection, batch_size=1, should_stack=True):
        movement_actions = self.movement_actions
        combo_actions = self.tensorflow_combo_actions
        indexer = tf.constant(1, dtype=tf.int32)
        action_selection = tf.cast(action_selection, tf.int32)
        if batch_size > 1:
            movement_actions = tf.expand_dims(movement_actions, 0)
            multiplier = tf.constant([int(batch_size), 1, 1])
            movement_actions = tf.tile(movement_actions, multiplier)
            combo_actions = tf.tile(tf.expand_dims(combo_actions, 0), multiplier)
            indexer = tf.constant(np.arange(0, batch_size, 1), dtype=tf.int32)
            yaw_actions = tf.squeeze(tf.slice(movement_actions, [0, 0, 0], [-1, 1, -1]))
            pitch_actions = tf.squeeze(tf.slice(movement_actions, [0, 1, 0], [-1, 1, -1]))
            roll_actions = tf.squeeze(tf.slice(movement_actions, [0, 2, 0], [-1, 1, -1]))
        else:
            yaw_actions = movement_actions[0]
            pitch_actions = movement_actions[1]
            roll_actions = movement_actions[2]

        # we get the options based on each individual index in the batches.  so this returns batch_size options
        steer = tf.gather_nd(yaw_actions, tf.stack([indexer, action_selection[0]], axis=1))
        pitch = tf.gather_nd(pitch_actions, tf.stack([indexer, action_selection[1]], axis=1))
        roll = tf.gather_nd(roll_actions, tf.stack([indexer, action_selection[2]], axis=1))

        button_combo = tf.gather_nd(combo_actions, tf.stack([indexer, action_selection[3]], axis=1))
        new_shape = [len(self.combo_list), batch_size]
        button_combo = tf.reshape(button_combo, new_shape)
        throttle = button_combo[0]
        jump = button_combo[1]
        boost = button_combo[2]
        handbrake = button_combo[3]
        controller_option = [throttle, steer, pitch, steer, roll, jump, boost, handbrake]
        controller_option = [tf.cast(option, tf.float32) for option in controller_option]
        # print(controller_option)
        if should_stack:
            return tf.stack(controller_option, axis=1)
        return controller_option

    def get_random_option(self):
        return [random.randrange(5), random.randrange(5), random.randrange(5), random.randrange(24)]

    def run_func_on_split_tensors(self, input_tensors, split_func, return_as_list=False):
        """
        Optionally splits the tensor and runs a function on the split tensor
        If the tensor should not be split it runs the function on the entire tensor
        :param input_tensors: needs to have shape of (?, num_actions)
        :param split_func: a function that is called with a tensor or array the same rank as input_tensor.
            It should return a tensor with the same rank as input_tensor
        :param return_as_list If true then the result will be a list of tensors instead of a single stacked tensor
        :return: a stacked tensor (see tf.stack) or the same tensor depending on if it is in split mode or not.
        """

        if not isinstance(input_tensors, collections.Sequence):
            input_tensors = [input_tensors]

        total_input = []
        for i in self.action_sizes:
            total_input.append([])

        for tensor in input_tensors:
            total_action_size = 0
            for i, val in enumerate(self.action_sizes):
                starting_length = len(total_input[i])
                if isinstance(tensor, collections.Sequence):
                    if len(tensor) == self.get_logit_size():
                        # grabs each slice of tensor
                        total_input[i].append(tensor[total_action_size:total_action_size + val])
                    else:
                        total_input[i].append(tensor[i])
                else:
                    if len(tensor.get_shape()) == 0:
                        total_input[i].append(tf.identity(tensor, name='copy' + str(i)))
                    elif tensor.get_shape()[0] == self.get_logit_size():
                        total_input[i].append(tf.slice(tensor, [total_action_size], [val]))
                    elif tensor.get_shape()[1] == self.get_logit_size():
                        total_input[i].append(tf.slice(tensor, [0, total_action_size], [-1, val]))
                    elif tensor.get_shape()[1] == self.get_number_actions():
                        total_input[i].append(tf.slice(tensor, [0, i], [-1, 1]))
                    elif tensor.get_shape()[1] == 1:
                        total_input[i].append(tf.identity(tensor, name='copy' + str(i)))
                total_action_size += val
                if starting_length == len(total_input[i]):
                    print('tensor ignored', tensor)

        if len(total_input[0]) != len(input_tensors):
            print('mis match in tensor length')

        results = []
        for i, val in enumerate(self.action_list_names):
            with tf.name_scope(val):
                try:
                    functional_input = total_input[i]
                    results.append(split_func(*functional_input))
                except Exception as e:
                    print('exception in split func ', val)
                    raise e

        if return_as_list:
            return results

        return tf.stack(results, axis=1)

    def optionally_split_numpy_arrays(self, numpy_array, split_func, is_already_split=False):
        """
        Optionally splits the tensor and runs a function on the split tensor
        If the tensor should not be split it runs the function on the entire tensor
        :param numpy_array: needs to have shape of (?, num_actions)
        :param split_func: a function that is called with a tensor the same rank as input_tensor.
            It should return a tensor with the same rank as input_tensor
        :param is_already_split: True if the array is already sliced by action length
        :return: a stacked tensor (see tf.stack) or the same tensor depending on if it is in split mode or not.
        """

        total_input = []
        total_action_size = 0
        for i, val in enumerate(self.action_sizes):
            if not is_already_split:
                total_input.append(numpy_array[:, total_action_size:val])
            else:
                total_input.append(numpy_array[i])
            total_action_size += val

        result = []
        for element in total_input:
            result.append(split_func(element))

        return result

    def create_action_indexes_graph(self, real_action, batch_size=None):
        #slice each index
        throttle = tf.slice(real_action, [0, 0], [-1, 1])
        steer = tf.slice(real_action, [0, 1], [-1, 1])
        pitch = tf.slice(real_action, [0, 2], [-1, 1])
        yaw = tf.slice(real_action, [0, 3], [-1, 1])
        roll = tf.slice(real_action, [0, 4], [-1, 1])
        jump = tf.slice(real_action, [0, 5], [-1, 1])
        boost = tf.slice(real_action, [0, 6], [-1, 1])
        handbrake = tf.slice(real_action, [0, 7], [-1, 1])

        conditional = tf.logical_and(tf.not_equal(steer, yaw), tf.less(tf.abs(steer), tf.abs(yaw)))
        use_yaw = tf.cast(conditional, tf.float32)
        use_steer = 1.0 - use_yaw
        steer = use_yaw * yaw + use_steer * steer

        steer_index = self._find_closet_real_number_graph(steer)
        pitch_index = self._find_closet_real_number_graph(pitch)
        roll_index = self._find_closet_real_number_graph(roll)

        rounded_throttle = tf.maximum(-1.0, tf.minimum(1.0, tf.round(throttle * 1.5)))

        # throttle, jump, boost, handbrake -> index number
        binary_combo_index = tf.squeeze(tf.constant(16.0) * tf.cast(tf.equal(rounded_throttle, 1), tf.float32) +
                                        tf.constant(8.0) * tf.cast(tf.equal(rounded_throttle, 0), tf.float32) +
                                        tf.constant(4.0) * jump +
                                        tf.constant(2.0) * boost +
                                        tf.constant(1.0) * handbrake)

        result = tf.stack([steer_index, pitch_index, roll_index, binary_combo_index], axis=1)
        return result
