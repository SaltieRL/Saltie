# MIT License
#
# Copyright (c) 2018 LHolten@Github Hytak#5125@Discord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from rlbot.agents.base_agent import SimpleControllerState
from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter
from agents.swarm.teacher_agent import SwarmAgent


class LeviSwarmAgent(SwarmAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        import torch
        self.torch = torch
        self.empty_controller = SimpleControllerState()

    def get_manager_path(self):
        return os.path.join(path, 'examples', 'levi', 'torch_manager.py')

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    def get_output(self, packet):
        """
        Predicts an output given the input
        :param packet: The game_tick_packet
        :return:
        """
        if not packet.game_info.is_round_active:
            return self.empty_controller
        if packet.game_cars[self.index].is_demolished:
            return self.empty_controller

        arr = self.input_formatter.create_input_array([packet], batch_size=1)

        assert (arr[0].shape == (1, 3, 9))
        assert (arr[1].shape == (1, 5))

        output = self.advanced_step(arr)

        return self.output_formatter.format_model_output(output, [packet], batch_size=1)[0]

    def advanced_step(self, arr):
        arr = [self.torch.from_numpy(x).float() for x in arr]

        with self.torch.no_grad():
            output = self.model.forward(*arr)
        return output
