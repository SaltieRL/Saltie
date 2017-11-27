import numpy as np
import itertools
import sys

class OptionMap:
    option_map = dict()

    def __init__(self, options):
        for i in range(len(options)):
            self.add_option(i, options[i])

    def add_option(self, index, option):
        tupleOption = tuple(np.array(option, dtype=np.float32))
        self.option_map[tupleOption] = index

    def has_key(self, option):
        tupleOption = tuple(np.array(option, dtype=np.float32))
        return tupleOption in self.option_map
    def get_key(self, option):
        tupleOption = tuple(np.array(option, dtype=np.float32))
        return self.option_map[tupleOption]


def createOptions():
    """
    Creates all variations of all of the options.
    :return: A combination of all options. This is an array of an array
    """
    throttle = np.arange(-1, 2, 1)
    steer = np.arange(-1, 2, 1)
    pitch = np.arange(-1, 2, 1)
    yaw = np.arange(-1, 2, 1)
    roll = np.arange(-1, 2, 1)
    jump = [True, False]
    boost = [True, False]
    handbrake = [True, False]
    option_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    print(option_list)
    entirelist = list(itertools.product(*option_list))
    print(len(entirelist))
    return entirelist

def compare_options(option1, option2):
    loss = 0
    for i in range(len(option1)):
        loss += abs(option1[i] - option2[i])
    return loss

def find_matching_option(optionMap, possibleOptions, real_option):
    # first time we do a close match I guess
    if optionMap.has_key(real_option):
        #print('found a matching object!')
        return real_option, optionMap.get_key(real_option), 0
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
