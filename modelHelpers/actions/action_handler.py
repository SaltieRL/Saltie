import collections
import itertools
import numpy as np
import random
import sys
import tensorflow as tf


class ActionMap:
    action_map = dict()

    def __init__(self, actions):
        for i in range(len(actions)):
            self.add_action(i, actions[i])

    def add_action(self, index, action):
        tupleaction = tuple(np.array(action, dtype=np.float32))
        self.action_map[tupleaction] = index

    def has_key(self, action):
        tupleaction = tuple(np.array(action, dtype=np.float32))
        return tupleaction in self.action_map
    def get_key(self, action):
        tupleaction = tuple(np.array(action, dtype=np.float32))
        return self.action_map[tupleaction]


class ActionHandler:
    range_size = 5
    action_list_names = ['actions']
    control_names = ['throttle', 'steer', 'pitch', 'steer', 'roll', 'jump', 'boost', 'handbrake']

    def __init__(self, split_mode=False):
        self.split_mode = split_mode
        self.actions = self.create_actions()
        self.action_map = ActionMap(self.actions)

        self.actions_split, self.split_action_sizes = self.create_actions_split()
        self.action_map_split = ActionMap(self.actions_split[3])

    def is_split_mode(self):
        return self.split_mode

    def get_split_sizes(self):
        return self.split_action_sizes

    def get_action_size(self):
        """
        :return: the size of the logits layer in a model
        """
        if not self.split_mode:
            return len(self.actions)

        counter = 0
        for action in self.actions_split:
            counter += len(action)
        return counter

    def create_actions(self):
        """
        Creates all variations of all of the actions.
        :return: A combination of all actions. This is an array of an array
        """
        throttle = np.arange(-1, 2, 1)
        steer = np.arange(-1, 2, 1)
        pitch = np.arange(-1, 2, 1)
        yaw = np.arange(-1, 2, 1)
        roll = np.arange(-1, 2, 1)
        jump = [True, False]
        boost = [True, False]
        handbrake = [True, False]
        action_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        entirelist = list(itertools.product(*action_list))
        return entirelist

    def create_actions_split(self):
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
        button_combo = list(itertools.product(*action_list))
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
        return actions, split_actions_sizes

    def create_controller_output_from_actions(self, action_selection):
        if len(action_selection) != len(self.actions_split):
            print('ACTION SELECTION IS NOT THE SAME LENGTH returning invalid action data')
            return [0, 0, 0, 0, 0, False, False, False]
        steer = self.actions_split[0][action_selection[0]]
        pitch = self.actions_split[1][action_selection[1]]
        roll = self.actions_split[2][action_selection[2]]
        button_combo = self.actions_split[3][action_selection[3]]
        throttle = button_combo[0]
        jump = button_combo[1]
        boost = button_combo[2]
        handbrake = button_combo[3]
        controller_option = [throttle, steer, pitch, steer, roll, jump, boost, handbrake]
        # print(controller_option)
        return controller_option

    def create_tensorflow_controller_output_from_actions(self, action_selection, batch_size=1):
        movement_actions = self.movement_actions
        combo_actions = self.combo_actions
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
        new_shape = [self.action_list_size, batch_size]
        button_combo = tf.reshape(button_combo, new_shape)
        throttle = button_combo[0]
        jump = button_combo[1]
        boost = button_combo[2]
        handbrake = button_combo[3]
        controller_option = [throttle, steer, pitch, steer, roll, jump, boost, handbrake]
        controller_option = [tf.cast(option, tf.float32) for option in controller_option]
        # print(controller_option)
        return tf.stack(controller_option, axis=1)

    def create_action_label(self, real_action):
        if self.split_mode:
            indexes = self._create_split_indexes(real_action)
            return self._create_split_label(indexes)
        index = self._find_matching_action(real_action)
        return self._create_one_hot_encoding(index)

    def create_action_index(self, real_action):
        if self.split_mode:
            return self._create_split_indexes(real_action)
        return self._find_matching_action(real_action)

    def _create_split_indexes(self, real_action):
        steer = real_action[1]
        yaw = real_action[3]
        if steer != yaw and abs(steer) < abs(yaw):
            # only take the larger magnitude number
            steer = yaw

        steer_index = self._find_closet_real_number(steer)
        pitch_index = self._find_closet_real_number(real_action[2])
        roll_index = self._find_closet_real_number(real_action[4])
        button_combo = self.action_map_split.get_key([round(real_action[0]), real_action[5], real_action[6], real_action[7]])

        return [steer_index, pitch_index, roll_index, button_combo]

    def _create_split_label(self, action_indexes):
        encoding = np.zeros(39)
        encoding[action_indexes[0] + 0] = 1
        encoding[action_indexes[1] + 5] = 1
        encoding[action_indexes[2] + 10] = 1
        encoding[action_indexes[3] + 15] = 1
        return encoding

    def _find_closet_real_number(self, number):
        if number <= -0.25:
            if number <= -0.75:
                return 0
            else:
                return 1
        elif number < 0.75:
            if number < 0.25:
                return 2
            else:
                return 3
        else:
            return 4

    def _compare_actions(self, action1, action2):
        loss = 0
        for i in range(len(action1)):
            loss += abs(action1[i] - action2[i])
        return loss

    def _find_matching_action(self, real_action):
        # first time we do a close match I guess
        if self.action_map.has_key(real_action):
            #print('found a matching object!')
            return self.action_map.get_key(real_action)
        closest_action = None
        index_of_action = 0
        counter = 0
        current_loss = sys.float_info.max
        for action in self.actions:
            loss = self._compare_actions(action, real_action)
            if loss < current_loss:
                current_loss = loss
                closest_action = action
                index_of_action = counter
            counter += 1
        return index_of_action

    def _create_one_hot_encoding(self, index):
        array = np.zeros(self.get_action_size())
        array[index] = 1
        return array

    def create_model_output(self, logits):
        return self.run_func_on_split_tensors(logits,
                                              lambda input_tensor: tf.argmax(input_tensor, 1))

    def create_controller_from_selection(self, selection):
        if self.split_mode:
            return self.create_controller_output_from_actions(selection)
        else:
            return self.actions[selection]

    def get_random_action(self):
        pass

    def get_random_option(self):
        if self.split_mode:
            return [random.randrange(5), random.randrange(5), random.randrange(5), random.randrange(24)]
        return random.randrange(self.get_action_size())
        pass

    def run_func_on_split_tensors(self, input_tensors, split_func, return_as_list = False):
        """
        Optionally splits the tensor and runs a function on the split tensor
        If the tensor should not be split it runs the function on the entire tensor
        :param tf: tensorflow
        :param input_tensors: needs to have shape of (?, num_actions)
        :param split_func: a function that is called with a tensor or array the same rank as input_tensor.
            It should return a tensor with the same rank as input_tensor
        :return: a stacked tensor (see tf.stack) or the same tensor depending on if it is in split mode or not.
        """

        if not isinstance(input_tensors, collections.Sequence):
            input_tensors = [input_tensors]

        if not self.split_mode:
            return split_func(*input_tensors)

        total_input = []
        for i in self.split_action_sizes:
            total_input.append([])

        for tensor in input_tensors:
            total_action_size = 0
            for i, val in enumerate(self.split_action_sizes):
                if isinstance(tensor, collections.Sequence):
                    if len(tensor) == self.get_action_size():
                        # grabs each slice of tensor
                        total_input[i].append(tensor[total_action_size:total_action_size + val])
                    else:
                        total_input[i].append(tensor[i])
                else:
                    if len(tensor.get_shape()) == 0:
                        total_input[i].append(tf.identity(tensor, name='copy' + str(i)))
                    elif tensor.get_shape()[0] == self.get_action_size():
                        total_input[i].append(tf.slice(tensor, [total_action_size], [val]))
                    elif tensor.get_shape()[1] == self.get_action_size():
                        total_input[i].append(tf.slice(tensor, [0, total_action_size], [-1, val]))
                    elif tensor.get_shape()[1] == 4:
                        total_input[i].append(tf.slice(tensor, [0, i], [-1, 1]))
                    elif tensor.get_shape()[1] == 1:
                        total_input[i].append(tf.identity(tensor, name='copy' + str(i)))
                total_action_size += val

        if len(total_input[0]) != len(input_tensors):
            print('mis match in tensor length')

        results = []
        for i, val in enumerate(self.action_list_names):
            with tf.name_scope(val):
                try:
                    results.append(split_func(*total_input[i]))
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
        if not self.split_mode:
            return split_func(numpy_array)

        total_input = []
        total_action_size = 0
        for i, val in enumerate(self.split_action_sizes):
            if not is_already_split:
                total_input.append(numpy_array[:, total_action_size:val])
            else:
                total_input.append(numpy_array[i])
            total_action_size += val

        result = []
        for element in total_input:
            result.append(split_func(element))

        return result

    def get_cross_entropy_with_logits(self, labels, logits, name):
        """
        In split mode there can be more than one class at a time.
        This is so that
        :param tf:
        :param labels:
        :param logits:
        :param name:
        :return:
        """
        if self.split_mode:
            return tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32), logits=logits, name=name+'s')
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name=name + 'ns')

    def _find_closet_real_number_graph(self, number):
        pure_number = tf.round(number * 2.0) / 2.0
        comparison = tf.Variable(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), dtype=tf.float32)
        pure_number = tf.cast(pure_number, tf.float32)
        equal = tf.equal(comparison, pure_number)
        index = tf.argmax(tf.cast(equal, tf.float32), axis=1)
        return tf.cast(index, tf.float32)

    def create_indexes_graph(self, real_action):
        if self.split_mode:
            return self._create_split_indexes_graph(real_action)
        else:
            raise Exception

    def _create_split_indexes_graph(self, real_action):
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
