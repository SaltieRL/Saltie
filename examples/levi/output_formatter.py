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


# symmetric:
# throttle - scalar
# pitch - scalar
# boost - boolean
# handbrake - boolean
# start_jump - boolean
# end_jump - boolean
# double_jump - boolean
# flip - boolean
# flip_forward - scalar

# asymmetric:
# flip_sideways - scalar
# steer - scalar
# yaw - scalar
# roll - scalar

class LeviOutputFormatter:
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.jump = False

    @staticmethod
    def format_numpy_output(new_controller_state: SimpleControllerState) -> ndarray:
        double = new_controller_state.yaw == 0.0 and new_controller_state.pitch == 0.0

        result = np.array([[
            new_controller_state.throttle,
            new_controller_state.pitch,
            1 if new_controller_state.boost else -1,
            1 if new_controller_state.handbrake else -1,
            1 if new_controller_state.jump else -1,  # start_jump
            1 if not new_controller_state.jump else -1,  # end_jump
            1 if new_controller_state.jump and double else -1,  # double_jump
            1 if new_controller_state.jump and not double else -1,  # flip
            new_controller_state.pitch,  # flip_forward

            new_controller_state.yaw,  # flip_sideways
            new_controller_state.steer,
            new_controller_state.yaw,
            new_controller_state.roll,
        ]])

        return result

    def get_mask(self, packet: GameTickPacket) -> ndarray:
        on_ground = packet.game_cars[self.index].has_wheel_contact
        can_double_jump = not packet.game_cars[self.index].double_jumped and not on_ground
        boost_available = not packet.game_cars[self.index].boost == 0

        mask = np.array([[
            1,  # throttle
            1 if not on_ground else 0,  # pitch
            1 if boost_available else 0,  # boost
            1 if on_ground else 0,  # handbrake
            1 if not self.jump and on_ground else 0,  # start_jump
            1 if self.jump and can_double_jump else 0,  # end_jump
            1 if not self.jump and can_double_jump else 0,  # double_jump
            1 if not self.jump and can_double_jump else 0,  # flip
            1 if not self.jump and can_double_jump else 0,  # flip_forward

            1 if not self.jump and can_double_jump else 0,  # flip_sideways
            1 if on_ground else 0,  # steer
            1 if not on_ground else 0,  # yaw
            1 if not on_ground else 0,  # roll
        ]])

        return mask

    def get_mask_supervised(self, packet: GameTickPacket, controller) -> ndarray:
        on_ground = packet.game_cars[self.index].has_wheel_contact
        can_double_jump = not packet.game_cars[self.index].double_jumped and not on_ground
        boost_available = not packet.game_cars[self.index].boost == 0
        flipping = (controller.yaw != 0.0 or controller.pitch != 0.0) and can_double_jump and controller.jump

        mask = np.array([[
            1 if not controller.boost else 0,  # throttle
            1 if not on_ground else 0,  # pitch
            1 if boost_available else 0,  # boost
            1 if on_ground else 0,  # handbrake
            1 if not self.jump and on_ground else 0,  # start_jump
            1 if self.jump and can_double_jump else 0,  # end_jump
            1 if not self.jump and can_double_jump else 0,  # double_jump
            1 if not self.jump and can_double_jump else 0,  # flip
            1 if flipping else 0,  # flip_forward

            1 if flipping else 0,  # flip_sideways
            1 if on_ground else 0,  # steer
            1 if not on_ground and not flipping else 0,  # yaw
            1 if not on_ground and not flipping else 0,  # roll
        ]])

        return mask

    def format_controller_output(self, action, packet: GameTickPacket) -> SimpleControllerState:
        controller_state = SimpleControllerState()

        controller_state.throttle = action[0].item()
        controller_state.boost = action[2].item() > semi_random(3)
        controller_state.handbrake = action[3].item() > semi_random(3)

        pitch = action[1].item()
        yaw = action[11].item()
        roll = action[12].item()

        can_jump = packet.game_cars[self.index].has_wheel_contact
        can_double_jump = not packet.game_cars[self.index].double_jumped and not can_jump
        jumping = self.jump and can_double_jump

        jump = jumping
        if not jumping and can_jump and action[4].item() > semi_random(5):  # start_jump
            jump = True
        if jumping and not can_jump and action[5].item() > semi_random(3):  # end_jump
            jump = False
        if not jumping and can_double_jump:
            if action[6].item() > semi_random(3):  # double_jump
                jump = True
                pitch = 0
                yaw = 0
                roll = 0
                # print("double")
            elif action[7].item() > semi_random(3):  # flip
                jump = True
                pitch = action[8].item()  # flip_forward
                yaw = action[9].item()  # flip_sideways
                roll = 0
                # print("flip")

        controller_state.pitch = pitch
        controller_state.jump = jump
        controller_state.steer = action[10].item()
        controller_state.yaw = yaw
        controller_state.roll = roll

        # we should not reference the new_controller_state, because it can change
        # we only need jump, so we can just copy the value
        self.jump = jump

        return controller_state

    @staticmethod
    def get_model_output_dimension():
        return 13,


def semi_random(power):
    # return pow(random() - random(), power)
    return 0
