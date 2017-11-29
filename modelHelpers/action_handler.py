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

    def __init__(self, split_mode = False):
        self.split_mode = split_mode
        self.actions = self.create_actions()
        self.action_map = ActionMap(self.actions)

        self.actions_split = self.create_actions_split()
        self.action_map_split = ActionMap(self.actions_split[3])

    def is_split_mode(self):
        return self.split_mode

    def get_action_size(self):
        """
        :param split_mode: True if we should use the reduced action size
        :return: the size of the logits layer in a model
        """
        if self.split_mode:
            return 39
        return len(self.actions)

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
        :return: A combination of all actions. This is an array of an array
        """

        steer = np.arange(-1, 1.5, .5)
        pitch = np.arange(-1, 1.5, .5)
        roll = np.arange(-1, 1.5, .5)
        throttle = np.arange(-1, 2, 1)
        jump = [True, False]
        boost = [True, False]
        handbrake = [True, False]
        action_list = [throttle, jump, boost, handbrake]
        # 24 + 5 + 5 + 5 = 39
        button_combo = list(itertools.product(*action_list))
        actions = []
        actions.append(steer)
        actions.append(pitch)
        actions.append(roll)
        actions.append(button_combo)
        return actions

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
        throttle = real_action[0]
        steer = real_action[1]
        pitch = real_action[2]
        yaw = real_action[3]
        roll = real_action[4]
        jump = real_action[5]
        boost = real_action[6]
        handbrake = real_action[7]
        if steer != yaw:
            # only take the larger magnitude number
            if abs(steer) < abs(yaw):
                steer = yaw

        steer_index = self._find_closet_real_number(steer)
        pitch_index = self._find_closet_real_number(pitch)
        roll_index = self._find_closet_real_number(roll)
        button_combo = self.action_map_split.get_key([throttle, jump, boost, handbrake])

        return [steer_index, pitch_index, roll_index, button_combo]

    def _create_split_label(self, action_indexes):
        encoding = np.zeros(39)
        encoding[action_indexes[0] + 0] = 1
        encoding[action_indexes[1] + 5] = 1
        encoding[action_indexes[2] + 10] = 1
        encoding[action_indexes[3] + 15] = 1
        return encoding

    def _find_closet_real_number(self, number):
        if abs(-1 - number) <= abs(-0.5 - number):
            return 0
        if abs(-0.5 - number) <= abs(0.0 - number):
            return 1
        if abs(0.0 - number) <= abs(0.5 - number):
            return 2
        if abs(0.5 - number) <= abs(1 - number):
            return 3
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

        output1 = []
        output2 = []
        output3 = []
        output4 = []

        i = 0
        for tensor in input_tensors:
            i+=1
            if isinstance(tensor, collections.Sequence):
                output1.append(tensor[0])
                output2.append(tensor[1])
                output3.append(tensor[2])
                output4.append(tensor[3])
            else:
                if tensor.get_shape()[1] == self.get_action_size():
                    output1.append(tf.slice(tensor, [0, 0], [-1, self.range_size]))
                    output2.append(tf.slice(tensor, [0, self.range_size], [-1, self.range_size]))
                    output3.append(tf.slice(tensor, [0, self.range_size * 2], [-1, self.range_size]))
                    output4.append(tf.slice(tensor, [0, self.range_size * 3], [-1, 24]))
                elif tensor.get_shape()[1] == 4:
                    output1.append(tf.slice(tensor, [0, 0], [-1, 1]))
                    output2.append(tf.slice(tensor, [0, 1], [-1, 1]))
                    output3.append(tf.slice(tensor, [0, 2], [-1, 1]))
                    output4.append(tf.slice(tensor, [0, 3], [-1, 1]))
                elif tensor.get_shape()[1] == 1:
                    output1.append(tf.identity(tensor, name='copy1'))
                    output2.append(tf.identity(tensor, name='copy2'))
                    output3.append(tf.identity(tensor, name='copy3'))
                    output4.append(tf.identity(tensor, name='copy4'))

        result1 = split_func(*output1)
        result2 = split_func(*output2)
        result3 = split_func(*output3)
        result4 = split_func(*output4)

        if return_as_list:
            return [result1, result2, result3, result4]

        return tf.stack([result1, result2, result3, result4], axis=1)

    def optionally_split_numpy_arrays(self, numpy_array, split_func, is_already_split=False):
        """
        Optionally splits the tensor and runs a function on the split tensor
        If the tensor should not be split it runs the function on the entire tensor
        :param numpy_array: needs to have shape of (?, num_actions)
        :param split_func: a function that is called with a tensor the same rank as input_tensor.
            It should return a tensor with the same rank as input_tensor
        :return: a stacked tensor (see tf.stack) or the same tensor depending on if it is in split mode or not.
        """
        if not self.split_mode:
            return split_func(numpy_array)

        if not is_already_split:
            output1 = numpy_array[:, 0:5]
            output2 = numpy_array[:, 5:10]
            output3 = numpy_array[:, 10:15]
            output4 = numpy_array[:, 15:]
        else:
            output1 = numpy_array[0]
            output2 = numpy_array[1]
            output3 = numpy_array[2]
            output4 = numpy_array[3]

        result1 = split_func(output1)
        result2 = split_func(output2)
        result3 = split_func(output3)
        result4 = split_func(output4)

        return [result1, result2, result3, result4]

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
