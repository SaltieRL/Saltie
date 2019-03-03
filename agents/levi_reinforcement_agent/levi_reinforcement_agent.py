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
import math
from agents.swarm.swarm_agent import SwarmAgent

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter
import numpy as np
from rlbot.utils.game_state_util import GameState, GameInfoState
from rlbot.agents.base_agent import SimpleControllerState


class LeviReinforcementTeacherAgent(SwarmAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        import torch
        self.torch = torch
        self.blue = None
        self.orange = None

        from agents.human_teacher.controller import HytakControllerInput
        self.controller_input = HytakControllerInput()

        self.oops = False
        self.data_dict = {
                'spatial': np.zeros((1, 3, 9), float),
                'extra': np.zeros((1, 5), float),
                'action': np.zeros((1, 13), float),
                'mask': np.zeros((1, 13), bool),
                'next_spatial': np.zeros((1, 3, 9), float),
                'next_extra': np.zeros((1, 5), float),
                'reward': np.zeros(1, float),
                'end': np.zeros(1, bool),
            }

        self.data_ready = False

    def get_manager_path(self):
        return os.path.join(path, 'examples', 'levi', 'levi_reinforcement_manager.py')

    def get_output(self, packet):
        """
        Predicts an output given the input
        :param packet: The game_tick_packet
        :return:
        """
        rigid = self.get_rigid_body_tick()
        arr = self.input_formatter.get_input_from_rigid(rigid)

        spatial_x = arr[0][0, 0]
        spatial_y = arr[0][0, 1]
        spatial_z = arr[0][0, 2]
        loss = (spatial_x[0] - spatial_x[1]) ** 2 + \
               (spatial_y[0] - spatial_y[1]) ** 2 + \
               (spatial_z[0] - spatial_z[1]) ** 2

        self.data_dict['reward'][0] = -math.sqrt(loss)

        self.data_dict['next_spatial'][:] = arr[0][:]
        self.data_dict['next_extra'][:] = arr[1][:]

        if not packet.game_info.is_round_active:
            blue, orange = total_goals(packet)
            if blue != self.blue and self.blue is not None:
                # self.data_dict['reward'][0] = 1 if self.team == 0 else -1
                self.data_dict['end'][0] = 1
                self.record()
            if orange != self.orange and self.orange is not None:
                # self.data_dict['reward'][0] = 1 if self.team == 1 else -1
                self.data_dict['end'][0] = 1
                self.record()

            self.blue, self.orange = blue, orange
            self.data_ready = False
            return self.empty_controller

        if packet.game_cars[self.index].is_demolished:
            self.data_ready = False
            return self.empty_controller

        self.data_dict['end'][0] = 0
        self.record()

        with self.torch.no_grad():
            tensors = [self.torch.from_numpy(x).float() for x in arr]
            assert (tensors[0].size() == (1, 3, 9))
            assert (tensors[1].size() == (1, 5))
            out_tensors = self.model.forward(*tensors)
            new_output, _, _, _ = (x.numpy() for x in out_tensors)

        new_output = np.random.normal(new_output, 1).clip(-1, 1)

        self.render()

        # game_info_state = GameInfoState(game_speed=1.5)
        # game_state = GameState(game_info=game_info_state)
        # self.set_game_state(game_state)

        mask = self.output_formatter.get_mask(packet)
        mask[0, 4] = 0  # no jumping
        mask[0, 3] = 0  # no handbrake
        mask[0, 0] = 0  # not throttle
        assert (mask.shape == (1, 13))

        controls = self.output_formatter.format_controller_output(new_output[0] * mask[0], packet)
        controls.jump = False
        controls.throttle = 1
        controls.handbrake = False

        self.data_dict['spatial'][:] = arr[0].copy()[:]
        self.data_dict['extra'][:] = arr[1].copy()[:]
        self.data_dict['action'][:] = self.output_formatter.format_numpy_output(rigid.players[self.index].input)[:]
        self.data_dict['mask'][:] = mask[:]

        self.data_ready = True

        if self.controller_input != self.empty_controller:
            return self.controller_input

        return controls

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    def record(self):
        try:
            if self.data_ready:
                self.data_ready = False
                self.game_memory.record(self.data_dict)
        except:
            if not self.oops:
                print('something went wrong')
                self.oops = True

    def render(self):
        self.renderer.begin_rendering()
        y = 80 if self.team else 0
        red = self.renderer.white()
        # self.renderer.draw_string_2d(0, y, 2, 2,
        #                              f"value = {round(value[0, 0].item(), 2)} -> ", red)
        self.renderer.draw_string_2d(0, y + 40, 2, 2,
                                     f"reward = {round(self.data_dict['reward'][0].item(), 2)}", red)
        self.renderer.end_rendering()


def total_goals(packet):
    blue = 0
    orange = 0
    for car in range(packet.num_cars):
        if packet.game_cars[car].team == 0:
            blue += packet.game_cars[car].score_info.goals
        else:
            orange += packet.game_cars[car].score_info.goals

    return blue, orange
