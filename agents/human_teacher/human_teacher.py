import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.agents.base_agent import SimpleControllerState

from agents.swarm.swarm_agent import SwarmAgent

from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter


class HumanTeacher(SwarmAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        # sys.path.insert(0, path)  # this is for separate process imports
        from agents.human_teacher.controller import HytakControllerInput
        self.controller_input = HytakControllerInput()
        self.teacher_formatter = self.create_output_formatter()
        self.empty_controller = SimpleControllerState()
        import torch
        self.torch = torch

    def get_manager_path(self):
        return os.path.join(path, 'examples', 'levi', 'torch_manager.py')

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    def get_output(self, game_tick_packet: GameTickPacket):
        if not game_tick_packet.game_info.is_round_active:
            return self.empty_controller
        if game_tick_packet.game_cars[self.index].is_demolished:
            return self.empty_controller

        arr = self.input_formatter.create_input_array([game_tick_packet], batch_size=1)

        if self.controller_input == self.empty_controller:
            output = self.advanced_step(arr)
            return self.output_formatter.format_model_output(output, [game_tick_packet])[0]

        teacher_output = self.controller_input
        teacher_output, mask = self.teacher_formatter.format_numpy_output(teacher_output, game_tick_packet)

        assert (arr[0].shape == (1, 3, 9))
        assert (arr[1].shape == (1, 5))
        assert (teacher_output.shape == (1, 13))
        assert (mask.shape == (1, 13))

        # return self.controller_input
        output = self.advanced_step(arr).numpy()
        output = output*0.5 + teacher_output*0.5

        self.game_memory.append(arr, output, mask)

        return self.output_formatter.format_model_output(output, [game_tick_packet])[0]

    def advanced_step(self, arr):
        arr = [self.torch.from_numpy(x).float() for x in arr]

        with self.torch.no_grad():
            output = self.model.forward(*arr)
        return output
