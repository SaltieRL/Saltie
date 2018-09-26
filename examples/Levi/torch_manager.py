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

from swarm_trainer.base_hive_manager import BaseHiveManager
from quicktracer import trace


class TorchManager(BaseHiveManager):
    optimizer = None
    loss_function = None

    def __init__(self, agent_metadata_queue, quit_event):
        import torch
        self.torch = torch  # before initialization as this is needed for `setup_trainer`
        super().__init__(agent_metadata_queue, quit_event)

    def setup_trainer(self):
        self.optimizer = self.torch.optim.Adamax(self.actor_model.parameters())
        self.loss_function = self.torch.nn.L1Loss()

    def get_model(self):
        from examples.levi.torch_model import SymmetricModel
        return SymmetricModel()

    def get_shared_model_handle(self):
        return self.actor_model.share_memory()

    def initialize_training(self, load_model=False, load_exp=False):
        if load_model:
            file_path = self.get_file_path()
            self.actor_model.load_state_dict(self.torch.load(file_path))
        if load_exp:
            file_path = self.get_file_path()  # should be different actually
            self.game_memory.load(file_path)

    def train_step(self, formatted_input, formatted_output, rewards=None, batch_size=1):
        self.optimizer.zero_grad()

        formatted_input = [self.torch.from_numpy(x).float() for x in formatted_input]
        formatted_output = self.torch.from_numpy(formatted_output).float()

        network_output = self.actor_model.forward(*formatted_input)

        loss = self.loss_function(network_output, formatted_output)
        loss.backward()
        trace(loss.item())

        self.optimizer.step()

    def finish_training(self, save_model=False, save_exp=False):
        if save_model:
            file_path = self.get_file_path()
            print('saving model at:', file_path)
            self.torch.save(self.actor_model.state_dict(), file_path)
        if save_exp:
            self.game_memory.save(self.actor_model.get_model_name() + '.exp')
