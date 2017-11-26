import numpy as np
import itertools
import sys

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

def compare_options(option1, option2):
    loss = 0
    for i in range(len(option1)):
        loss += abs(option1[i] - option2[i])
    return loss

def find_matching_option(possibleOptions, real_option):
    # first time we do a close match I guess
    closest_option = None
    index_of_option = 0
    counter = 0
    current_loss = sys.float_info.max
    for option in possibleOptions:
        loss = compare_options(option, real_option)
        if loss < current_loss:
            current_loss = loss
            closest_option = option
            index_of_option = counter
        counter += 1
    return closest_option, index_of_option, current_loss
