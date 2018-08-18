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

from random import random
from rlbot.utils.structures.bot_input_struct import PlayerInput
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LeviOutputFormatter(BaseOutputFormatter):
    player_input = PlayerInput()

    def __init__(self, index):
        super().__init__()
        self.index = index

    def format_model_output(self, action, packet, batch_size=1):
        in_the_air = packet.game_cars[self.index].jumped

        self.player_input.throttle = action[0]
        self.player_input.pitch = action[1]
        self.player_input.boost = action[2] > semi_random(3)
        self.player_input.handbrake = action[3] > semi_random(3)

        action_1 = action[4] > semi_random(5)
        action_2 = action[5] > semi_random(5)

        jump_on_ground = not self.player_input.jump and not in_the_air and action_1
        flip_in_air = not self.player_input.jump and action_2
        jump_in_air = in_the_air and (flip_in_air or not action_2) and (action_1 or action_2)

        self.player_input.jump = jump_on_ground or jump_in_air

        self.player_input.roll = action[6]
        self.player_input.steer = action[7]
        self.player_input.yaw = action[8]

        return self.player_input

    def get_model_output_dimension(self):
        return [(9,)]


def semi_random(power):
    return pow(random() - random(), power)
