import numpy as np
import itertools


def createOptions():
    """
    Creates all variations of all of the options.
    :return: A combination of all options. This is an array of an array
    """
    throttle = np.arange(-1, 1, 1)
    steer = np.arange(-1, 1, 1)
    pitch = np.arange(-1, 1, 1)
    yaw = np.arange(-1, 1, 1)
    roll = np.arange(-1, 1, 1)
    jump = [True, False]
    boost = [True, False]
    handbrake = [True, False]
    option_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    return list(itertools.product(*option_list))

