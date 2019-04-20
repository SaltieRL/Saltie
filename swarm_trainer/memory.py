import numpy as np
from multiprocessing import Lock


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
        self.lock = Lock()

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

        self.size = final_size

    @non_parallel
    def get_sample(self, size: int) -> dict:
        size = min(size, self.size)
        indices = np.unique(np.random.randint(self.size, size=size))

        for key in self.data_dict.keys():
            self.sample[key].resize((indices.size,) + self.data_dict[key].shape[1:], refcheck=False)
            self.sample[key][:] = self.data_dict[key][indices].copy()

        return self.sample

    def get_size(self) -> int:
        return self.size

    @non_parallel
    def save(self, file: str) -> None:
        import torch
        print('trying')
        torch.save((self.data_dict, self.size, self.limit), file)
        print('done')

    @non_parallel
    def load(self, file: str) -> None:
        import torch
        self.data_dict, self.size, self.limit = torch.load(file)
