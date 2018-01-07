import itertools

import numpy as np
import tensorflow as tf

from modelHelpers.actions.action_handler import ActionHandler


current_scheme = [[('steer', (-1, 1.5, .5)), ('pitch', (-1, 1.5, .5)), ('roll', (-1, 1.5, .5))],
                  [('throttle', (-1, 2, 1)), ('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                  [('yaw', 'steer')]]

combo = 'combo'

class DynamicActionHandler(ActionHandler):
    """Very dynamic for controls and splitting.
        Assumes everything is in tensorflow
    """

    control_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
    # rules, tuples mean they take the same spot
    ranged_actions = []
    action_list_names = []
    action_name_index_map = {}
    combo_name_index_map = {}
    action_sizes = []
    indexed_controls = []
    combo_list = []
    button_combo = []

    def __init__(self, control_scheme):
        self.control_scheme = control_scheme
        super().__init__(False)
        pass

    def create_range_action(self, item):
        action_data = np.arange(*item[1])
        return action_data

    def create_actions(self):
        ranges = self.control_scheme[0]
        combo_scheme = self.control_scheme[1]
        copies = self.control_scheme[2]

        for item in ranges:
            action = self.create_range_action(item)
            self.action_sizes.append(len(action))
            self.action_name_index_map[item[0]] = len(self.action_list_names)
            self.action_list_names.append(item[0])
            self.indexed_controls.append(action)

        self.ranged_actions = self.indexed_controls[:]

        for item in combo_scheme:
            action = self.create_range_action(item)
            self.action_name_index_map[item[0]] = combo
            self.combo_name_index_map[item[0]] = len(self.combo_list)
            self.combo_list.append(action)
        self.button_combo = list(itertools.product(*self.combo_list))
        self.action_sizes.append(len(self.button_combo))
        self.action_name_index_map[combo] = len(self.action_list_names)
        self.action_list_names.append(combo)
        self.indexed_controls.append(self.button_combo)

        for item in copies:
            self.action_name_index_map[item[0]] = self.action_name_index_map[item[1]]
        return self.button_combo

    def create_actions_split(self):
        return self.indexed_controls, self.action_list_names

    def create_controller_output_from_actions(self, action_selection):
        if len(action_selection) != len(self.actions_split):
            print('ACTION SELECTION IS NOT THE SAME LENGTH returning invalid action data')
            return [0, 0, 0, 0, 0, False, False, False]

        controller_output = []
        for control in self.control_names:
            index = self.action_name_index_map[control]
            if index == combo:
                combo_index = self.action_name_index_map[combo]
                true_index = self.combo_name_index_map[control]
                controller_output.append(self.indexed_controls[combo_index][action_selection[combo_index]][true_index])
                continue
            controller_output.append(self.indexed_controls[index][action_selection[index]])
        # print(controller_output)
        return controller_output

    def create_tensorflow_controller_output_from_actions(self, action_selection, batch_size=1):
        controller_output = []

        ranged_actions = []
        combo_actions = tf.constant(np.transpose(np.array(self.button_combo)))
        action_selection = tf.cast(action_selection, tf.int32)

        # handle ranged actions
        multiplier = tf.constant([int(batch_size), 1])
        for ranged_action in self.ranged_actions:
            action = tf.constant(np.array(ranged_action))
            if batch_size > 1:
                expanded_action = tf.expand_dims(action, 0)
                action = tf.tile(expanded_action, multiplier)
            ranged_actions.append(action)

        # handle combo buttons
        if batch_size > 1:
            multiplier = tf.constant([1, int(batch_size), 1])
            combo_actions = tf.tile(tf.expand_dims(combo_actions, 1), multiplier)
            indexer = tf.constant(np.arange(0, batch_size, 1), dtype=tf.int32)
        else:
            indexer = tf.constant(1, dtype=tf.int32)

        # actually decoding the controls now the startup is done
        for control in self.control_names:
            index = self.action_name_index_map[control]
            if index == combo:
                combo_index = self.action_name_index_map[combo]
                true_index = self.combo_name_index_map[control]

                single_element = combo_actions[true_index]

                controller_output.append(
                    tf.gather_nd(single_element, tf.stack([indexer, action_selection[combo_index]], axis=1)))
                continue
            output = tf.gather_nd(ranged_actions[index], tf.stack([indexer, action_selection[index]], axis=1))
            controller_output.append(output)

        # make sure everything is the same type
        controller_output = [tf.cast(option, tf.float32) for option in controller_output]

        return tf.stack(controller_output, axis=1)

if __name__ == '__main__':
    action_handler = DynamicActionHandler(current_scheme)

    print(action_handler)
