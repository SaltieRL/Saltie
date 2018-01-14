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
    combo_list = []
    action_sizes = []
    actions = []
    action_list_names = ['actions']
    control_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
    control_size = len(control_names)

    def __init__(self):
        self.actions = self.create_actions()
        self.action_map = self.create_action_map()

    def reset(self):
        self.control_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
        self.control_size = len(self.control_names)

    def create_action_map(self):
        return ActionMap(self.actions)

    def is_split_mode(self):
        return False

    def get_action_sizes(self):
        """
        :return: A list of the sizes of each action
        """
        return [len(self.actions)]

    def get_number_actions(self):
        """
        :return: How many different actions there are
        """
        return 1

    def get_logit_size(self):
        """
        :return: the size of the logits layer in a model
        """
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
        jump = [False, True]
        boost = [False, True]
        handbrake = [False, True]
        action_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        self.action_sizes = [len(action) for action in action_list]
        self.combo_list = action_list
        entirelist = list(itertools.product(*action_list))
        return entirelist

    def create_controller_from_selection(self, action_selection):
        return self.actions[action_selection[0]]

    def create_tensorflow_controller_from_selection(self, action_selection, batch_size=1, should_stack=True):
        combo_actions = self.actions
        indexer = tf.constant(1, dtype=tf.int32)
        action_selection = tf.cast(action_selection, tf.int32)
        if batch_size > 1:
            multiplier = tf.constant([int(batch_size), 1, 1])
            combo_actions = tf.tile(tf.expand_dims(combo_actions, 0), multiplier)
            indexer = tf.constant(np.arange(0, batch_size, 1), dtype=tf.int32)

        button_combo = tf.gather_nd(combo_actions, tf.stack([indexer, action_selection[3]], axis=1))
        new_shape = [self.get_logit_size(), batch_size]
        button_combo = tf.reshape(button_combo, new_shape)
        controller_option = button_combo
        controller_option = [tf.cast(option, tf.float32) for option in controller_option]
        # print(controller_option)
        if should_stack:
            return tf.stack(controller_option, axis=1)
        return controller_option

    def create_action_label(self, real_action):
        index = self.create_action_index(real_action)
        return self._create_one_hot_encoding(index)

    def create_action_index(self, real_action):
        return [self._find_matching_action(real_action)]

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
        array = np.zeros(self.get_logit_size())
        array[index] = 1
        return array

    def create_model_output(self, logits):
        return self.run_func_on_split_tensors(logits,
                                              lambda input_tensor: tf.argmax(input_tensor, 1))

    def get_random_action(self):
        pass

    def get_random_option(self):
        return [random.randrange(self.get_logit_size())]

    def run_func_on_split_tensors(self, input_tensors, split_func):
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
        return split_func(*input_tensors)

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
        return split_func(numpy_array)

    def _find_closet_real_number_graph(self, number):
        pure_number = tf.round(number * 2.0) / 2.0
        comparison = tf.Variable(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), dtype=tf.float32)
        pure_number = tf.cast(pure_number, tf.float32)
        equal = tf.equal(comparison, pure_number)
        index = tf.argmax(tf.cast(equal, tf.float32), axis=1)
        return tf.cast(index, tf.float32)

    def round_action_graph(self, input, action_size):
        rounded_amount = float(action_size // 2)
        return tf.maximum(-1.0, tf.minimum(1.0, tf.round(input * rounded_amount) / rounded_amount))

    def _create_combo_index_graph(self, real_actions):
        binary_combo_index = tf.constant(0.0)
        for i, action_set in enumerate(reversed(self.combo_list)):
            powed = tf.constant(pow(2, i), dtype=tf.float32)
            action_taken = real_actions[i]
            if self.action_sizes[i] > 2:
                action_set = self.combo_list[i]
                new_range = self.action_sizes[i] // 2 + 1
                for j in range(new_range):
                    powed = tf.constant(pow(2, i + (new_range - 1 - j)), dtype=tf.float32)
                    binary_combo_index += powed * (tf.cast(
                        tf.equal(action_taken, action_set[len(action_set) - 1 - j]), tf.float32))
            else:
                binary_combo_index += powed * tf.cast(action_taken, tf.float32)
        return binary_combo_index

    def create_action_indexes_graph(self, real_action, batch_size=None):
        combo_list = []
        for i, control_set in enumerate(self.combo_list):
            bucketed_control = tf.slice(real_action, [0, i], [-1, 1])
            if len(control_set) > 2:
                bucketed_control = self.round_action_graph(bucketed_control, len(control_set))
            combo_list.append(bucketed_control)

        return self._create_combo_index_graph(combo_list)

    def get_action_loss_from_logits(self, logits, labels, index):
        """
        :param logits: A tensorflow logit
        :param labels: A label of what accured
        :param index: The index of the control in the actions list this maps to
        :return: The loss for this particular action
        """
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name=str(index) + 'ns')

    def scale_layer(self, layer, index):
        """
        Scales the layer if required
        :param layer: the output layer of the model
        :param index: The index regarding this specific action
        :return: A scaled layer
        """
        return layer

    def get_loss_type(self, index):
        return 'softmax'
