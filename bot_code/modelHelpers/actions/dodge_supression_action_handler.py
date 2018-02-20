from bot_code.modelHelpers.actions.dynamic_action_handler import DynamicActionHandler, COMBO
import tensorflow as tf

DODGE = 'dodge'


class DodgeActionHandler(DynamicActionHandler):

    dodge_suppressor_list = [['jump'], ['steer', 'pitch', 'roll', 'yaw']]

    def __init__(self, control_scheme):
        super().__init__(control_scheme)

    def reset(self):
        super().reset()

        # add doge as a control name
        # control size is still 7
        self.control_names = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake', DODGE]

    def create_controller_from_selection(self, action_selection):
        controller_output = super().create_controller_from_selection(action_selection)

        combo_index = self.action_name_index_map[COMBO]
        if self.should_suppress_dodge:
            suppressors = self.dodge_suppressor_list[0]
            should_suppress = self.actions[combo_index][action_selection[combo_index]][self.combo_name_index_map[DODGE]]
            for option in suppressors:
                should_suppress = should_suppress and controller_output[self.action_name_index_map[option]]
            if should_suppress:
                for suppressed_control in self.dodge_suppressor_list[1]:
                    controller_output[self.action_name_index_map[suppressed_control]] = 0.0
        # print(controller_output)
        return controller_output

    def create_tensorflow_controller_from_selection(self, action_selection, batch_size=1, should_stack=True):
        controller_output = super().create_tensorflow_controller_from_selection(action_selection, batch_size=batch_size,
                                                                                should_stack=False)

        should_suppress = controller_output[len(controller_output) - 1]
        for option in self.dodge_suppressor_list[0]:
            should_suppress = tf.logical_and(should_suppress,
                                             controller_output[self.control_names_index_map[option]])

        for suppressed_control in self.dodge_suppressor_list[1]:
            controller_output[self.control_names_index_map[suppressed_control]] *= tf.cast(should_suppress,
                                                                                           tf.float32)

        if should_stack:
            return tf.stack(controller_output, axis=1)
        return controller_output

    def _create_combo_index(self, real_action, combo_list):
        is_suppressed = True
        for option in self.dodge_suppressor_list[1]:
            is_suppressed = is_suppressed and self.round_action(
                real_action[self.control_names_index_map[option]], 7) == 0.0
        for option in self.dodge_suppressor_list[0]:
            is_suppressed = is_suppressed and self.round_action(
                real_action[self.control_names_index_map[option]], 7) == 1.0

        combo_list[self.combo_name_index_map[DODGE]] = is_suppressed
        return super()._create_combo_index(real_action, combo_list)

    def _create_combo_index_graph(self, combo_list, real_action=None):
        is_suppressed = tf.constant([True])
        for option in self.dodge_suppressor_list[1]:
            is_suppressed = tf.logical_and(is_suppressed, tf.equal(self.round_action_graph(
                real_action[self.control_names_index_map[option]], 7), 0.0))
        for option in self.dodge_suppressor_list[0]:
            is_suppressed = tf.logical_and(is_suppressed, tf.equal(self.round_action_graph(
                real_action[self.control_names_index_map[option]], 7), 1.0))

        combo_list[self.combo_name_index_map[DODGE]] = is_suppressed

        return super()._create_combo_index_graph(combo_list)
