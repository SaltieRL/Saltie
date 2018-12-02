import numpy as np
from multiprocessing import Lock


def non_parallel(func):
    def func_wrapper(self, *args, **kwargs):
        self.lock.acquire()
        result = func(self, *args, **kwargs)
        self.lock.release()
        return result
    return func_wrapper


class BaseMemory:
    def __init__(self, limit: int, shape_list: list):
        self.limit = limit
        self.size = 0
        self.data_list = [np.empty((limit,) + shape) for shape in shape_list]
        self.goal_list = {}
        self.lock = Lock()
        np.seterr(all='raise')

    @non_parallel
    def append(self, data_list: list):
        # space = self.limit - self.size
        size = data_list[0].shape[0]
        final_size = min(self.size + size, self.limit)
        data_fill = min(size, self.limit - self.size)
        data_random = size - data_fill
        indexes = np.random.randint(self.limit, size=data_random)

        for i, n in enumerate(self.data_list):
            n[self.size:final_size] = data_list[i][:data_fill]
            n[indexes] = data_list[i][data_fill:]

        self.size = final_size

    @non_parallel
    def get_sample(self, size: int) -> list:
        size = min(size, self.size)
        indexes = np.random.randint(self.size, size=size)

        data_list = [n[indexes].copy() for n in self.data_list]

        return data_list

    def set_goal(self, goal: int, time: float):
        self.goal_list[goal] = time

    def get_goal(self, goal: int) -> float:
        return self.goal_list.get(goal, None)

    def get_size(self):
        return self.size
