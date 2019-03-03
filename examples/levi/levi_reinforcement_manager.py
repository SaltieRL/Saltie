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
import numpy as np

eps = np.finfo(np.float32).eps.item()


class LeviReinforcementManager(BaseHiveManager):
    optimizer = None
    distance = None
    threshold = None
    smooth_l1 = None

    def __init__(self, agent_metadata_queue, quit_event, options):
        import torch
        self.torch = torch  # before initialization as this is needed for `setup_trainer`
        from torch import distributions
        self.dist = distributions
        from torch.nn import functional
        self.F = functional
        self.plot_count = 0
        self.step_count = 0

        super().__init__(agent_metadata_queue, quit_event, options)

    def setup_trainer(self):
        from torch.optim import Adadelta
        self.optimizer = Adadelta(self.model.parameters(), lr=0.7)
        from torch.nn.functional import pairwise_distance
        self.distance = pairwise_distance

    def get_shape_dict(self):
        action_shape = self.model.get_model_output_dimension()[0]
        state_shape_list = self.model.get_input_state_dimension()

        return {
            'spatial': state_shape_list[0],
            'extra': state_shape_list[1],
            'action': action_shape,
            'mask': action_shape,
            'next_spatial': state_shape_list[0],
            'next_extra': state_shape_list[1],
            'reward': (),
            'end': ()
        }

    def get_model(self):
        from examples.levi.torch_model import SymmetricModel
        return SymmetricModel()

    def get_shared_model_handle(self):
        return self.model.share_memory()

    def initialize_training(self, load_model=None, load_exp=None):
        if load_model:
            file_path = self.get_file_path()
            self.model.load_state_dict(self.torch.load(file_path))
        if load_exp:
            file_path = os.path.join(os.path.dirname(self.get_file_path()), 'memory.mem')
            self.game_memory.load(file_path)

    def train_step(self, data_dict: dict):
        Tensor = self.torch.Tensor

        spatial = self.torch.from_numpy(data_dict['spatial']).float()
        extra = self.torch.from_numpy(data_dict['extra']).float()
        taken_action = self.torch.from_numpy(data_dict['action']).float()
        mask = self.torch.from_numpy(data_dict['mask']).float()
        next_spatial = self.torch.from_numpy(data_dict['next_spatial']).float()
        next_extra = self.torch.from_numpy(data_dict['next_extra']).float()
        reward = self.torch.from_numpy(data_dict['reward']).float()
        end = self.torch.from_numpy(data_dict['end']).float()

        best_action, state_value, action_dist, value_dist = self.model.forward(spatial, extra)
        _, next_value, _, _ = self.model.forward(next_spatial, next_extra)

        action_value = reward * 0.02 + next_value * 0.98

        # calculating losses
        action_difference = self.distance(best_action.mul(mask), taken_action.mul(mask), p=1)

        value_loss = self.F.l1_loss(state_value, action_value, reduction='none')
        action_loss = action_difference + self.F.l1_loss(action_difference.detach(), action_dist, reduction='none')
        value_loss += self.F.l1_loss(value_loss.detach(), value_dist, reduction='none')

        # masking
        influence = self.dist.HalfNormal(action_dist, True).log_prob(action_difference).exp().detach()
        advantage = self.dist.Normal(state_value, value_dist, True).cdf(action_value).detach()

        # action stuff
        action_weight = advantage.float()
        action_weight /= action_weight.abs().mean().clamp(min=eps)

        # value stuff
        value_weight = influence.float()
        value_weight /= value_weight.mean().clamp(min=eps)

        # applying weights
        value_loss = value_loss.mul(value_weight)
        action_loss = action_loss.mul(action_weight)

        loss = (action_loss + value_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.plot_count += 1
        if self.plot_count % 1 == 0:
            self.plot_count = 0

            trace(value_loss.mean().item())
            trace(action_loss.mean().item())
            trace(state_value.mean().item())

    def finish_training(self, save_model=True, save_exp=True):
        if save_model:
            file_path = self.get_file_path()
            print('saving model at:', file_path)
            self.torch.save(self.model.state_dict(), file_path)
        if save_exp:
            file_path = os.path.join(os.path.dirname(self.get_file_path()), 'memory.mem')
            print('saving memory at:', file_path)
            self.game_memory.save(file_path)
