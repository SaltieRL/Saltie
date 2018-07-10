import os

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.botmanager.helper_process_request import HelperProcessRequest


class TorchLearner(BaseAgent):
    def __init__(self, name, team, index):
        from leviathan.output_formatter import OutputFormatter
        from leviathan.input_formatter import InputFormatter
        from leviathan.cool_atba import Atba
        import torch

        BaseAgent.__init__(self, name, team, index)
        self.pipe = None
        self.actor_model = None
        self.team_model = None
        self.game_memory = None
        self.atba = Atba()
        self.torch = torch
        self.output_formatter = OutputFormatter()
        self.input_formatter = InputFormatter(self.index, self.index)
        # self.input_formatter = InputFormatter(self.index, (self.index + 1) % 2)

    def get_helper_process_request(self) -> HelperProcessRequest:
        from torch.multiprocessing import Pipe

        file = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_manager.py'))
        key = 'hive_mind'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request

    def initialize_agent(self):
        from leviathan.torch_model import TeamModel
        self.actor_model = self.pipe.recv()
        self.game_memory = self.pipe.recv()
        self.team_model = TeamModel(self.team, self.actor_model)

    def get_output(self, game_tick_packet):
        spatial, car_stats = self.input_formatter.get_input(game_tick_packet)
        with self.torch.no_grad():
            action = self.team_model.forward_single(spatial, car_stats)

            desired_action = self.atba.forward(spatial, car_stats)
            compared_action = self.torch.unsqueeze(action, 0)
            desired_action = self.torch.unsqueeze(desired_action, 0)

            loss = self.torch.nn.functional.pairwise_distance(compared_action, desired_action, p=1).item()

        in_the_air = game_tick_packet.game_cars[self.index].jumped
        player_input = self.output_formatter.get_output(action, in_the_air)

        self.game_memory.append(spatial, car_stats, action, loss)

        # self.game_memory.append(spatial, car_stats, self.torch.zeros(9), 0)
        return player_input


if __name__ == '__main__':
    learner = TorchLearner('levi', 0, 0)