import random
import numpy as np
from multiprocessing import Lock


def non_parallel(func):
    def func_wrapper(self, *args, **kwargs):
        self.lock.acquire()
        result = func(self, *args, **kwargs)
        self.lock.release()
        return result
    return func_wrapper


class BaseRewardMemory:
    filled = False

    def __init__(self, limit, input_shape, output_shape):
        self.limit = limit
        self.input_data = [np.empty((0,) + shape) for shape in input_shape]
        self.action = np.empty((0,) + output_shape)
        # self.reward = np.array((0,))
        self.lock = Lock()
        np.seterr(all='raise')

    @non_parallel
    def append(self, input_data, action, reward=None):
        space = self.limit - self.action.shape[0]

        if action.shape[0] <= space:  # all of the new data fits
            self.input_data = [np.concatenate((n, input_data[i]), axis=0) for i, n in enumerate(self.input_data)]
            self.action = np.concatenate((self.action, action), axis=0)
        elif space == 0:  # none of the new data fits
            indexes = np.random.randint(self.action.shape[0], size=action.shape[0])
            for i, n in enumerate(self.input_data):
                n[indexes, :] = input_data[i]
            self.action[indexes, :] = action
        else:  # only part of the new data fits
            self.input_data = [np.concatenate((n, input_data[:space]), axis=0) for i, n in enumerate(self.input_data)]
            self.action = np.concatenate((self.action, action[:space]), axis=0)

            indexes = np.random.randint(self.action.shape[0], size=action.shape[0] - space)
            for i, n in enumerate(self.input_data):
                n[indexes, :] = input_data[i][space:]
            self.action[indexes, :] = action[space:]

        # if reward is not None:
        #     self.reward = np.concatenate((self.reward, reward), axis=0)

    @non_parallel
    def get_sample(self, amount):
        length = self.action.shape[0]

        if length <= amount:
            sample_input_data = [np.copy(n) for n in self.input_data]
            sample_action = np.copy(self.action)
            # sample_reward = np.copy(self.reward)

            return sample_input_data, sample_action, None

        i = random.randint(0, length - 1)
        j = i + amount

        if j > length:
            j %= length
            sample_input_data = [np.concatenate((n[i:], n[:j])) for n in self.input_data]
            sample_action = np.concatenate((self.action[i:], self.action[:j]), axis=0)

            # if self.reward.shape[0] > 0:
            #     sample_reward = np.concatenate((self.reward[i:], self.reward[:j]), axis=0)
            # else:
            #     sample_reward = np.copy(self.reward)
        else:
            sample_input_data = [np.copy(n[i:j]) for n in self.input_data]
            sample_action = np.copy(self.action[i:j])

            # if self.reward.shape[0] > 0:
            #     sample_reward = np.copy(self.reward[i:j])
            # else:
            #     sample_reward = np.copy(self.reward)

        return sample_input_data, sample_action, None

    @non_parallel
    def get_random_sample(self, amount):
        if self.action.shape[0] <= amount:
            return [n.copy() for n in self.input_data], self.action.copy(), None
        assert(self.action.shape[0] == self.input_data[0].shape[0])

        indexes = np.random.randint(self.action.shape[0], size=amount)

        sample_input_data = [n[indexes, :].copy() for n in self.input_data]
        sample_action = self.action[indexes, :].copy()

        return sample_input_data, sample_action, None

    @non_parallel
    def clear(self):
        self.input_data = [np.array([])]
        self.action = np.array((0, 9))
        # self.reward = np.array((0,))
