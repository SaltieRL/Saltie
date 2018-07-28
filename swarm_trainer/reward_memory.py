import random
import numpy as np
from multiprocessing import Lock
from framework.model_holder.base_model_holder import BaseModelHolder


class BaseRewardMemory:
    def __init__(self, model_holder: BaseModelHolder):
        self.model_holder = model_holder
        self.input_data = np.array([])
        self.action = np.array([])
        self.reward = np.array([])
        self.length = 0
        self.lock = Lock()

    def append(self, input_data, action, reward=None):
        self.lock.acquire()
        self.input_data = np.append(self.input_data, input_data)
        self.action = np.append(self.action, action)
        if reward is not None:
            self.reward = np.append(self.reward, reward)
        self.length += 1
        self.lock.release()

    def get_sample(self, amount):
        self.lock.acquire()

        if self.length <= amount:
            sample_input_data = np.copy(self.input_data)
            sample_action = np.copy(self.action)
            sample_reward = np.copy(self.reward)
            self.lock.release()

            return sample_input_data, sample_action, sample_reward

        i = random.randint(0, self.length - 1)
        j = i + amount

        if j > self.length:
            j %= self.length
            sample_input_data = np.concatenate((self.input_data[i:], self.input_data[:j]))
            sample_action = np.concatenate((self.action[i:], self.action[:j]))
            if len(self.reward) > 0:
                sample_reward = np.concatenate((self.reward[i:], self.reward[:j]))
            else:
                sample_reward = []
            self.lock.release()

            return sample_input_data, sample_action, sample_reward
        else:
            sample_input_data = np.copy(self.input_data[i:j])
            sample_action = np.copy(self.action[i:j])
            sample_reward = np.copy(self.reward[i:j])
            self.lock.release()

            return sample_input_data, sample_action, sample_reward

    def clear(self):
        self.input_data = np.array([])
        self.action = np.array([])
        self.reward = np.array([])
