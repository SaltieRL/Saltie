import numpy as np
from multiprocessing import Lock
import math


def non_parallel(func: callable) -> callable:
    def func_wrapper(self, *args, **kwargs):
        self.lock.acquire()
        result = func(self, *args, **kwargs)
        self.lock.release()
        return result
    return func_wrapper


class BaseMemory:
    def __init__(self, limit: int, shape_dict: dict) -> None:
        self.limit = limit
        self.size = 0
        self.data_dict = {key: np.zeros((limit,) + shape) for key, shape in shape_dict.items()}
        # self.data_dict['goal_time'] = np.zeros(limit)
        # self.data_dict['importance'] = np.zeros(limit)
        # self.data_dict['model_step'] = np.zeros(limit, np.uint32)
        self.lock = Lock()
        # self.time = 0
        # self.model_step = 0

        # self.previous_indices = None
        self.sample = {key: np.empty(()) for key in self.data_dict.keys()}
        np.seterr(all='raise')

    @non_parallel
    def record(self, data_dict: dict) -> None:
        size = next(iter(data_dict.values())).shape[0]
        final_size = min(self.size + size, self.limit)
        number_to_fill = min(size, self.limit - self.size)
        number_to_replace = size - number_to_fill

        indices_to_replace = np.random.randint(self.limit, size=number_to_replace)
        indices_to_fill = np.arange(self.limit)[self.size:final_size]
        indices = np.concatenate((indices_to_replace, indices_to_fill))

        for key in data_dict.keys():
            self.data_dict[key][indices] = data_dict[key]

        # self.data_dict['goal_time'][indices] = 0
        # self.data_dict['importance'][indices] = 1
        # self.data_dict['model_step'][indices] = self.model_step

        self.size = final_size

        # self.time = abs(data_dict['time']).max()

    @non_parallel
    def get_sample(self, size: int) -> dict:
        size = min(size, self.size)
        indices = np.unique(np.random.randint(self.size, size=size))

        for key in self.data_dict.keys():
            self.sample[key].resize((indices.size,) + self.data_dict[key].shape[1:], refcheck=False)
            self.sample[key][:] = self.data_dict[key][indices].copy()

        # self.sample['model_step'] = self.model_step - self.sample['model_step']
        # no_goal_mask = self.data_dict['goal_time'][indices] == 0
        # self.sample['goal_time'][no_goal_mask] = abs(self.sample['time'][no_goal_mask]) - self.time
        #
        # self.model_step += 1

        # self.previous_indices = indices
        return self.sample

    # @non_parallel
    # def goal(self, time: float) -> None:
    #     # negative time means orange goal/ car
    #
    #     all_goal_mask = self.data_dict['goal_time'] == 0  # zero because new goals
    #     goal_mask = (np.sign(self.data_dict['time']) == math.copysign(1, time)) & all_goal_mask
    #     own_goal_mask = (~goal_mask) & all_goal_mask
    #
    #     self.data_dict['goal_time'][goal_mask] = abs(self.data_dict['time'][goal_mask] - time)
    #     self.data_dict['goal_time'][own_goal_mask] = -abs(self.data_dict['time'][own_goal_mask] + time)

    # @non_parallel
    # def importance(self, importance) -> None:
    #     all_goal_mask = self.data_dict['goal_time'][self.previous_indices] != 0
    #     self.data_dict['importance'][self.previous_indices[all_goal_mask]] = abs(importance.numpy()[all_goal_mask])

    def get_size(self) -> int:
        return self.size

    @non_parallel
    def save(self, file: str) -> None:
        import torch
        # no_goal_mask = self.data_dict['goal_time'] == 0
        # self.data_dict['importance'][no_goal_mask] = 0
        print('trying')
        # torch.save((self.data_dict, self.size, self.limit, self.model_step), file)
        torch.save((self.data_dict, self.size, self.limit), file)
        print('done')

    @non_parallel
    def load(self, file: str) -> None:
        import torch
        # self.data_dict, self.size, self.limit, self.model_step = torch.load(file)
        self.data_dict, self.size, self.limit = torch.load(file)
