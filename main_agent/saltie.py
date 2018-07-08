import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from legacy.legacy_game_input_formatter import LegacyGameInputFormatter
from legacy.legacy_keras_model import LegacyKerasModel
from legacy.legacy_model_holder import LegacyModelHolder
from legacy.legacy_output_formatter import LegacyOutputFormatter


class Saltie(BaseAgent):

    model_holder = None

    def load_config(self):
        self.model_holder = LegacyModelHolder(LegacyKerasModel(),
                                         LegacyGameInputFormatter(self.team, self.index),
                                         LegacyOutputFormatter())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        return self.model_holder.predict(packet)
