from modelHelpers.actions.action_handler import ActionHandler
from modelHelpers.actions.dynamic_action_handler import DynamicActionHandler
from modelHelpers.actions.split_action_handler import SplitActionHandler

default_scheme = [[('steer', (-1, 1.5, .5)), ('pitch', (-1, 1.5, .5)), ('roll', (-1, 1.5, .5))],
                  [('throttle', (-1, 2, 1)), ('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                  [('yaw', 'steer')]]


def get_handler(split_mode, control_scheme):
    if not split_mode:
        return ActionHandler()
    if control_scheme == default_scheme:
        return SplitActionHandler()
    return DynamicActionHandler(control_scheme)
