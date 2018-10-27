import unittest
from .base_hive_manager import BaseHiveManager
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_setup(self):
        BaseHiveManager.setup_manager().Memory()

    def test_append(self):
        memory = BaseHiveManager.setup_manager().Memory()
        memory.append([np.zeros((10, 3, 9)), np.zeros((10, 5))], np.zeros((10, 9)))

    def test_get_random_sample(self):
        memory = BaseHiveManager.setup_manager().Memory()
        memory.append([np.zeros((10, 3, 9)), np.zeros((10, 5))], np.zeros((10, 9)))
        _, result, _ = memory.get_random_sample(10)

        assert(result.shape == (10, 9))


if __name__ == '__main__':
    unittest.main()
