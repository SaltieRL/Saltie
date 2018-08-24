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
from framework.output_formatter.base_output_formatter import BaseOutputFormatter
import numpy as np


class LeviOutputFormatter(BaseOutputFormatter):
    jumped = False

    def __init__(self, index):
        super().__init__()
        self.index = index

    def format_model_output(self, arr, batch_size=1):
        if batch_size != 1:
            raise NotImplementedError

        action = arr[0]
        packet = arr[1]  # hacky solution until we have packet support

        player_input = np.zeros(9)

        player_input[0] = action[0]  # throttle
        player_input[2] = action[1]  # pitch
        player_input[6] = action[2] > semi_random(3)  # boost
        player_input[7] = action[3] > semi_random(3)  # handbrake

        in_the_air = packet.game_cars[self.index].jumped
        jump_1 = action[4] > semi_random(5)
        jump_2 = action[5] > semi_random(5)

        jump_on_ground = not self.jumped and not in_the_air and jump_1
        flip_in_air = not self.jumped and jump_2
        jump_in_air = in_the_air and (flip_in_air or not jump_2) and (jump_1 or jump_2)

        player_input[5] = jump_on_ground or jump_in_air  # jump
        self.jumped = player_input[5]

        player_input[1] = action[6]  # steer
        player_input[3] = action[7]  # yaw
        player_input[4] = action[8]  # roll

        return [player_input]

    def get_model_output_dimension(self):
        return [(9,)]


def semi_random(power):
    return pow(random() - random(), power)
