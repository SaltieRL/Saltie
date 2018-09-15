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

from agents.swarm.swarm_agent import SwarmAgent
from examples.Levi.output_formatter import LeviOutputFormatter
from examples.Levi.input_formatter import LeviInputFormatter
import os
from rlbot.botmanager.helper_process_request import HelperProcessRequest


class LeviAgent(SwarmAgent):
    import torch
    pipe = None
    model = None
    input_formatter = None
    output_formatter = None
    game_memory = None

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_manager.py'))
        key = 'levi_hive_mind'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request

    def initialize_agent(self):
        self.model = self.pipe.recv()
        self.input_formatter = LeviInputFormatter(self.team, self.index)
        self.output_formatter = LeviOutputFormatter(self.index)
        self.game_memory = self.pipe.recv()

    def predict(self, packet):
        """
        Predicts an output given the input
        :param packet: The game_tick_packet
        :return:
        """
        arr = self.input_formatter.create_input_array([packet], batch_size=1)

        arr = [self.torch.from_numpy(x).float() for x in arr]

        with self.torch.no_grad():
            output = self.model.forward(*arr)
            self.game_memory.append(arr, output)  # should be replaced with hardcoded output

        output = [output[0], packet]

        return self.output_formatter.format_model_output(output, batch_size=1)[0]
