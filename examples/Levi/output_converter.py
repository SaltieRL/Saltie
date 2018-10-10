from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np
from numpy import ndarray


class OutputConverter:
    def __init__(self, index: int):
        self.index = index
        self.controller_state = SimpleControllerState()

    # SimpleControllerState
    def scs_to_numpy(self, new_controller_state: SimpleControllerState, packet: GameTickPacket) -> ndarray:
        result = np.array([
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
        ])
        self.controller_state = new_controller_state
        return result
