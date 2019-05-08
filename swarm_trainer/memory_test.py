import unittest
from .base_hive_manager import BaseHiveManager
import numpy as np
from .memory import BaseMemory
from multiprocessing.managers import BaseManager


class MyManager(BaseManager):
    pass


MyManager.register('Memory', BaseMemory)


class MyTestCase(unittest.TestCase):

    def test_setup(self):
        BaseHiveManager.setup_manager().Memory(3000, [(3, 9), (5,)], (13,))

    def test_append(self):
        memory = BaseMemory(2500, {"input_1": (3, 9), "input_2": (5,), "output": (13,)})
        for i in range(3):
            memory.record({"input_1": np.zeros((1000, 3, 9)), "input_2": np.zeros((1000, 5)), "output": np.zeros((1000, 13))})

    def test_get_random_sample(self):
        memory = BaseMemory(1000000, {"input_1": (3, 9), "input_2": (5,), "time": (1,)})
        for i in range(1001):
            memory.record({"input_1": np.zeros((1000, 3, 9)), "input_2": np.zeros((1000, 5)), "time": np.zeros((1000, 1))})
        for i in range(10000000):
            test = memory.get_sample(1000)

    def test_get_random_sample_2(self):
        manager = MyManager()
        manager.start()

        memory = manager.Memory(1000000, {"input_1": (3, 9), "input_2": (5,), "time": (1,)})
        input = {"input_1": np.zeros((1000, 3, 9)), "input_2": np.zeros((1000, 5)), "time": np.zeros((1000, 1))}

        for i in range(1001):
            input["input_1"][:] = np.zeros((1000, 3, 9))
            input["input_2"][:] = np.zeros((1000, 5))
            input["time"][:] = np.zeros((1000, 1))
            memory.record(input)
            print(memory.get_size())
        for i in range(10000000):
            test = memory.get_sample(1000)


if __name__ == '__main__':
    unittest.main()
