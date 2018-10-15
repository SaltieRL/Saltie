from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.logging_utils import get_logger
from rlbot.utils.structures.game_data_struct import GameTickPacket
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports
from examples.levi.input_formatter import LeviInputFormatter
from examples.levi.cool_atba import Atba
from examples.levi.output_formatter import LeviOutputFormatter


class CoolAtbaAgent(BaseAgent):
    input_formatter = None
    atba = None
    output_formatter = None

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.logger = get_logger(name)

    def initialize_agent(self):
        self.input_formatter = LeviInputFormatter(self.team, self.index)
        self.atba = Atba()
        self.output_formatter = LeviOutputFormatter(self.index)

    def get_output(self, game_tick_packet: GameTickPacket):
        arr = self.input_formatter.create_input_array([game_tick_packet])
        output = self.atba.get_action(arr)
        return self.output_formatter.format_model_output(output, game_tick_packet)[0]
