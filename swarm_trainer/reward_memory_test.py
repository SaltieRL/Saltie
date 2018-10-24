import unittest
from .base_hive_manager import BaseHiveManager
import numpy as np
from .reward_memory import BaseRewardMemory


class MyTestCase(unittest.TestCase):

    def test_setup(self):
        BaseHiveManager.setup_manager().Memory(3000, [(3, 9), (5,)], (9,))

    def test_append(self):
        memory = BaseRewardMemory(3000, [(3, 9), (5,)], (9,))
        memory.append([np.zeros((1000, 3, 9)), np.zeros((1000, 5))], np.zeros((1000, 9)))
        memory.append([np.zeros((1000, 3, 9)), np.zeros((1000, 5))], np.zeros((1000, 9)))
        memory.append([np.zeros((1000, 3, 9)), np.zeros((1000, 5))], np.zeros((1000, 9)))

    def test_get_random_sample(self):
        memory = BaseHiveManager.setup_manager().Memory(3000, [(3, 9), (5,)], (9,))
        memory.append([np.zeros((10, 3, 9)), np.zeros((10, 5))], np.zeros((10, 9)))
        _, result, _ = memory.get_random_sample(10)

        assert(result.shape == (10, 9))


if __name__ == '__main__':
    unittest.main()
