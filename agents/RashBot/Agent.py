import sys
import os

folder = os.path.dirname(os.path.realpath(__file__))
if folder not in sys.path:
    sys.path.append(folder)

from Bot import Process


class Agent:
    def __init__(self, name, team, index):
        self.index = index

    def get_output_vector(self, game):
        return Process(self, game)
