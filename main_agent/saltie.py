import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class Saltie(BaseAgent):

    model_holder = None

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

        from legacy.legacy_game_input_formatter import LegacyGameInputFormatter
        from legacy.legacy_keras_model import LegacyKerasModel
        from legacy.legacy_model_holder import LegacyModelHolder
        from legacy.legacy_output_formatter import LegacyOutputFormatter

        self.model_holder = LegacyModelHolder(LegacyKerasModel(),
                                              LegacyGameInputFormatter(self.team, self.index),
                                              LegacyOutputFormatter())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        result = self.model_holder.predict(packet)
        self.controller_state.throttle = result[0]
        self.controller_state.steer = result[1]
        self.controller_state.pitch = result[2]
        self.controller_state.yaw = result[3]
        self.controller_state.roll = result[4]
        self.controller_state.jump = result[5]
        self.controller_state.boost = result[6]
        self.controller_state.boost = result[7]

        return self.controller_state
