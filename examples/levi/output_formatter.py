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
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np
from numpy import ndarray


class LeviOutputFormatter:
    controller_state = SimpleControllerState()

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.controller_state = SimpleControllerState()

    def format_model_output(self, arr: ndarray, packet: GameTickPacket, batch_size=1) -> ndarray:
        if batch_size != 1:
            raise NotImplementedError
        action = arr[0]

        self.controller_state.throttle = action[0]
        self.controller_state.pitch = action[1]
        self.controller_state.boost = action[2] > semi_random(3)
        self.controller_state.handbrake = action[3] > semi_random(3)

        in_the_air = packet.game_cars[self.index].jumped
        jump_1 = action[4] > semi_random(5)
        jump_2 = action[5] > semi_random(5)

        jump_on_ground = not self.controller_state.jump and not in_the_air and jump_1
        flip_in_air = not self.controller_state.jump and jump_2
        jump_in_air = in_the_air and (flip_in_air or not jump_2) and (jump_1 or jump_2)

        self.controller_state.jump = jump_on_ground or jump_in_air

        self.controller_state.steer = action[6]
        self.controller_state.yaw = action[7]
        self.controller_state.roll = action[8]

        return np.array([self.controller_state])

    @staticmethod
    def get_model_output_dimension():
        return (9,)

    def format_numpy_output(self, new_controller_state: SimpleControllerState, packet: GameTickPacket) -> ndarray:
        result = np.array([[
            new_controller_state.throttle,
            new_controller_state.pitch,
            1 if new_controller_state.boost else -1,
            1 if new_controller_state.handbrake else -1,
            1 if new_controller_state.jump else -1,
            1 if (new_controller_state.jump and not self.controller_state.jump
                  and packet.game_cars[self.index].jumped) or packet.game_cars[self.index].double_jumped else -1,
            new_controller_state.steer,
            new_controller_state.yaw,
            new_controller_state.roll,
        ]])
        self.controller_state = new_controller_state
        return result


def semi_random(power):
    # return pow(random() - random(), power)
    return 0
