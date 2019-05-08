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
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent, BOT_CONFIG_AGENT_HEADER
from rlbot.parsing.custom_config import ConfigHeader, ConfigObject
from rlbot.utils.game_state_util import GameState, GameInfoState

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter


class LeviAgent(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        sys.path.insert(0, path)  # this is for separate process imports
        import torch
        self.torch = torch
        self.empty_controller = SimpleControllerState()
        self.model_path = None
        self.model = None
        self.input_formatter = None
        self.output_formatter = None

    def load_config(self, config_object_header: ConfigHeader):
        self.model_path = config_object_header.get('model_path')

    def initialize_agent(self):
        self.model = self.get_model()
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()
        self.model.load_state_dict(self.torch.load(self.get_file_path()), strict=False)

    def get_file_path(self):
        return os.path.join(path, self.model_path)

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

        arr = self.input_formatter.create_input_array([packet])

        with self.torch.no_grad():
            tensors = [self.torch.from_numpy(x).float() for x in arr]
            assert (tensors[0].size() == (1, 3, 9))
            assert (tensors[1].size() == (1, 5))
            out_tensors = self.model.forward(*tensors)
            new_output, _ = (x.numpy() for x in out_tensors)
            # new_output, _, _, _ = (x.numpy() for x in out_tensors)

        mask = self.output_formatter.get_mask(packet)
        assert (mask.shape == (1, 13))

        controls = self.output_formatter.format_controller_output(new_output[0] * mask[0], packet)

        # game_info_state = GameInfoState(game_speed=3.0)
        # game_state = GameState(game_info=game_info_state)
        # self.set_game_state(game_state)

        return controls

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    @staticmethod
    def get_model():
        from examples.levi.torch_model import SymmetricModel
        return SymmetricModel()

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        super(LeviAgent, LeviAgent).create_agent_configurations(config)
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('model_path', str, default=os.path.join('models', 'cool_atba.mdl'),
                         description='Path to the model file')

    # def visualize_net(self, spatial, extra):
    #     actor = self.model.actor
    #     spatial = spatial[0, :, :].squeeze()
    #
    #     weight_x = actor.input_x.multiplier.weight
    #
    #     # result_x = spatial[:, 0, :2].cross(weight_x.t()[0, :2])
    #     result_x = weight_x[:, :2].clone()
    #     result_x[:, 0], result_x[:, 1] = weight_x[:, 1], weight_x[:, 0].neg()
    #     result_x = (result_x * 100).renorm(2, 1, 1)
    #
    #     if self.team == 1:
    #         result_x *= -1
    #         # vectors[0:2] *= -1
    #         # pos[0:2] *= -1
    #
    #     self.renderer.begin_rendering()
    #     # for l in range(-1000, 1000, 100):
    #     self.renderer.draw_line_3d([spatial[0, 0], spatial[1, 0], spatial[2, 0]],
    #                                [spatial[0, 1], spatial[1, 1], spatial[2, 1]],
    #                                self.renderer.black())
    #     # + l * result_x[0, 0]
    #     # + l * result_x[0, 1]
    #
    #     self.renderer.end_rendering()
