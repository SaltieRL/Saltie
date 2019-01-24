import numpy as np
from multiprocessing import Lock
import math
import random


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
        self.data_dict['status'] = np.zeros(limit, bool)
        self.data_dict['importance'] = np.ones(limit)
        self.lock = Lock()
        self.time = 0

        self.previous_indices = None
        self.sample = {key: np.empty(()) for key in self.data_dict.keys()}
        np.seterr(all='raise')

    @non_parallel
    def record(self, data_dict: dict) -> None:
        # space = self.limit - self.size
        size = data_dict['time'].shape[0]
        final_size = min(self.size + size, self.limit)
        number_to_fill = min(size, self.limit - self.size)
        number_to_replace = size - number_to_fill

        # probability = np.exp(np.negative(self.data_dict['importance']))
        # probability = probability / probability.sum()
        # indices_to_replace = np.random.choice(np.arange(self.limit), size=number_to_replace, p=probability)

        indices_to_replace = np.random.randint(self.limit, size=number_to_replace)
        # if random.random() > 0.1 and number_to_replace == 1:
        #     indices_to_replace = np.array([np.argmin(self.data_dict['importance'])])
        indices_to_fill = np.arange(self.limit)[self.size:final_size]
        indices = np.concatenate((indices_to_replace, indices_to_fill))

        for key in data_dict.keys():
            # print(key)
            # print(self.data_dict[key].shape)
            self.data_dict[key][indices] = data_dict[key]
            # print(self.data_dict[key].shape)

        # print(self.data_dict['importance'][indices])
        # print(indices)

        self.data_dict['status'][indices] = 0
        self.data_dict['importance'][indices] = 1

        self.size = final_size

        new_time = abs(data_dict['time'].max())
        if new_time > self.time:
            self.time = new_time

    @non_parallel
    def get_sample(self, size: int) -> dict:
        size = min(size, self.size)
        indices = np.unique(np.random.randint(self.size, size=size))

        for key in self.data_dict.keys():
            self.sample[key].resize((indices.size,) + self.data_dict[key].shape[1:], refcheck=False)
            self.sample[key][:] = self.data_dict[key][indices].copy()

        time_mask = ~self.data_dict['status'][indices]
        self.sample['time'][time_mask] = abs(self.sample['time'][time_mask]) - self.time
        # print(data_dict['time'][time_mask])

        self.previous_indices = indices
        return self.sample

    @non_parallel
    def goal(self, time: float) -> None:
        # negative time means orange goal/ car

        time_mask = ~self.data_dict['status']
        # masked_time = self.data_dict['time'][time_mask]
        goal_mask = (np.sign(self.data_dict['time']) == math.copysign(1, time)) & time_mask
        own_goal_mask = (~goal_mask) & time_mask

        self.data_dict['time'][goal_mask] = abs(self.data_dict['time'][goal_mask] - time)
        self.data_dict['time'][own_goal_mask] = -abs(self.data_dict['time'][own_goal_mask] + time)
        self.data_dict['status'][time_mask] = 1

        # print(self.data_status[goal_mask])

    @non_parallel
    def importance(self, importance) -> None:
        mask = self.data_dict['status'][self.previous_indices]
        self.data_dict['importance'][self.previous_indices[mask]] = abs(importance.numpy()[mask])

        # print(importance)

    def get_size(self) -> int:
        return self.size

    @non_parallel
    def save(self, file: str) -> None:
        import torch
        time_mask = ~self.data_dict['status']
        # worst case scenario reward
        self.data_dict['time'][time_mask] = abs(self.data_dict['time'][time_mask]) - self.time
        self.data_dict['status'][time_mask] = 1
        print('trying')
        torch.save((self.data_dict, self.size, self.limit, self.time), file)
        print('done')

    @non_parallel
    def load(self, file: str) -> None:
        import torch
        self.data_dict, self.size, self.limit, self.time = torch.load(file)
