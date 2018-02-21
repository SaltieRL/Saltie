from bot_code.modelHelpers.actions.action_handler import ActionHandler
from bot_code.modelHelpers.actions.dynamic_action_handler import DynamicActionHandler, LOSS_SQUARE_MEAN, LOSS_SPARSE_CROSS, \
    LOSS_ABSOLUTE_DIFFERENCE
from bot_code.modelHelpers.actions.split_action_handler import SplitActionHandler

from collections import namedtuple

'''
ranges:
  A range is a configuration object which holds information about its name, possible values and loss function.
  Currently represented as a tuple:
  - Name (string)
  - Arguments to np.arange (tuple of (startInclusive, endExclusive, step))
  - Optional: The loss function (one of the LOSS_* constants)
combo_scheme:
  Seems to be similar to ranges. TODO: figure out how this works in more detail. Loss functions seem to never be specified.
copies:
  A `copy` is a tuple of two names where the actions of the latter (name) is forwarded to the former.
'''
ControlScheme = namedtuple('ControlScheme', 'ranges combo_scheme copies')


THROTTLE = 'throttle'
STEER = 'steer'
YAW = 'yaw'
PITCH = 'pitch'
ROLL = 'roll'
JUMP = 'jump'
BOOST = 'boost'
HANDBRAKE = 'handbrake'


default_scheme = ControlScheme(
  ranges=[(STEER, (-1, 1.5, .5)), (PITCH, (-1, 1.5, .5)), (ROLL, (-1, 1.5, .5))],
  combo_scheme=[(THROTTLE, (-1, 2, 1)), (JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1))],
  copies=[(YAW, STEER)],
)

super_split_scheme = ControlScheme(
  ranges=[(THROTTLE, (-1, 1.5, .5)), (STEER, (-1, 1.5, .5)), (YAW, (-1, 1.5, .5)), (PITCH, (-1, 1.5, .5)), (ROLL, (-1, 1.5, .5))],
  combo_scheme=[(JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1))],
  copies=[],
)

super_split_scheme_no_combo = ControlScheme(
  ranges=[(THROTTLE, (-1, 1.25, .25)), (STEER, (-1, 1.25, .25)), (YAW, (-1, 1.25, .25)), (PITCH, (-1, 1.25, .25)), (ROLL, (-1, 1.25, .25)),(JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1))],
  combo_scheme=[],
  copies=[],
)

only_steer_split_scheme = ControlScheme(
  ranges=[(STEER, (-1, 1.5, .5))],
  combo_scheme=[(THROTTLE, (-1, 2, 1)), (JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1)), (YAW, (-1, 2, 1)), (PITCH, (-1, 2, 1)), (ROLL, (-1, 2, 1))],
  copies=[],
)

regression_controls = ControlScheme(
  ranges=[
    (THROTTLE,   (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (STEER,      (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (YAW,        (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (PITCH,      (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (ROLL,       (-1, 1.5, .5), LOSS_SQUARE_MEAN)],
  combo_scheme=[(JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1))],
  copies=[],
)

regression_controls_no_combo = ControlScheme(
  ranges=[
    (THROTTLE,   (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (STEER,      (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (YAW,        (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (PITCH,      (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (ROLL,       (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (JUMP, (0, 2, 1)),
    (BOOST, (0, 2, 1)),
    (HANDBRAKE, (0, 2, 1))],
  combo_scheme=[],
  copies=[],
)

mixed_controls = ControlScheme(
  ranges=[
    (THROTTLE, (-1, 1.5, .5), LOSS_SPARSE_CROSS),
    (STEER,    (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE),
    (YAW,      (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE),
    (PITCH,    (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE),
    (ROLL,     (-1, 1.5, .5), LOSS_ABSOLUTE_DIFFERENCE)
  ],
  combo_scheme=[(JUMP, (0, 2, 1)), (BOOST, (0, 2, 1)), (HANDBRAKE, (0, 2, 1))],
  copies=[],
)

regression_everything = ControlScheme(
  ranges=[
    (THROTTLE,  (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (STEER,     (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (YAW,       (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (PITCH,     (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (ROLL,      (-1, 1.5, .5), LOSS_SQUARE_MEAN),
    (JUMP,      (0, 2, 1),     LOSS_SQUARE_MEAN),
    (BOOST,     (0, 2, 1),     LOSS_SQUARE_MEAN),
    (HANDBRAKE, (0, 2, 1),     LOSS_SQUARE_MEAN)],
  combo_scheme=[],
  copies=[],
)


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
