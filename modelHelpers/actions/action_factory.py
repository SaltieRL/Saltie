from modelHelpers.actions.action_handler import ActionHandler
from modelHelpers.actions.dynamic_action_handler import DynamicActionHandler, LOSS_SQUARE_MEAN, LOSS_SPARSE_CROSS, \
    LOSS_ABSOLUTE_DIFFERENCE
from modelHelpers.actions.split_action_handler import SplitActionHandler

default_scheme = [[('steer', (-1, 1.5, .5)), ('pitch', (-1, 1.5, .5)), ('roll', (-1, 1.5, .5))],
                  [('throttle', (-1, 2, 1)), ('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                  [('yaw', 'steer')]]

super_split_scheme = [[('throttle', (-1, 1.5, .5)), ('steer', (-1, 1.5, .5)),
                       ('yaw', (-1, 1.5, .5)), ('pitch', (-1, 1.5, .5)), ('roll', (-1, 1.5, .5))],
                      [('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                      []]

only_steer_split_scheme = [[('steer', (-1, 1.5, .5))],
                           [('throttle', (-1, 2, 1)), ('jump', (0, 2, 1)), ('boost', (0, 2, 1)),
                            ('handbrake', (0, 2, 1)), ('yaw', (-1, 2, 1)),
                            ('pitch', (-1, 2, 1)), ('roll', (-1, 2, 1))],
                           []]

regression_controls = [[('throttle', (-1, 1.5, .5), LOSS_SQUARE_MEAN), ('steer', (-1, 1.5, .5), LOSS_SQUARE_MEAN),
                        ('yaw', (-1, 1.5, .5), LOSS_SQUARE_MEAN), ('pitch', (-1, 1.5, .5), LOSS_SQUARE_MEAN),
                        ('roll', (-1, 1.5, .5), LOSS_SQUARE_MEAN)],
                       [('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                       []]

mixed_controls = [[('throttle', (-1, 1.5, .5), LOSS_SPARSE_CROSS), ('steer', (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE),
                        ('yaw', (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE), ('pitch', (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE),
                        ('roll', (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE)],
                       [('jump', (0, 2, 1)), ('boost', (0, 2, 1)), ('handbrake', (0, 2, 1))],
                       []]

def get_handler(split_mode=True, control_scheme=default_scheme):
    """
    Creates a handler based on the options given.
    This defaults to returning SplitActionHandler
    :param split_mode: If False control_scheme is ignored.
                       False means that everything is rolled up into a single action
    :param control_scheme:  A dynamic control scheme,  if it is the default scheme an optimized split handler is used.
    :return: A handler that handles actions
    """
    if not split_mode:
        return ActionHandler()
    if control_scheme == default_scheme:
        return SplitActionHandler()
    return DynamicActionHandler(control_scheme)
