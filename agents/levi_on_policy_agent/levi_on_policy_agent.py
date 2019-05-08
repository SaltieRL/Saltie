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

import os
import sys
from pynput.keyboard import Listener, Key
from agents.swarm.teacher_agent import TeacherAgent

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter
from rlbot.botmanager.helper_process_request import HelperProcessRequest
from rlbot.utils.game_state_util import GameState, GameInfoState
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent, BOT_CONFIG_AGENT_HEADER
from rlbot.parsing.custom_config import ConfigHeader, ConfigObject
from rlbot.utils.logging_utils import get_logger
from framework.utils import get_repo_directory
from quicktracer import trace


class LeviReinforcementTeacherAgent(BaseAgent):
    pipe = None
    queue = None
    model = None
    input_formatter = None
    output_formatter = None
    optimizer = None

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports
        self.logger = get_logger(name)

        import torch
        self.torch = torch

        self.manager_path = None
        self.model_path = None
        self.load_model = None

        self.blue = None
        self.orange = None
        self.empty_controller = SimpleControllerState()

        self.offset = torch.zeros(1, 13)
        self.theta = 0.9
        self.sigma = 0.002

        self.score = 5

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(f'runs/bot_neg_0.2')

    def initialize_agent(self):
        self.model = self.pipe.recv()
        self.queue = self.pipe.recv()

        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()
        self.optimizer = self.torch.optim.Adadelta(self.model.parameters())
        file_path = os.path.join(get_repo_directory(), f'models/opt{self.team}.sd')
        if os.path.isfile(file_path):
            self.optimizer.load_state_dict(self.torch.load(file_path))
        for group in self.optimizer.param_groups:
            group['lr'] = 0.2

    @staticmethod
    def get_manager_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'levi_on_policy_manager.py')

    def get_output(self, packet):
        """
        Predicts an output given the input
        :param packet: The game_tick_packet
        :return:
        """
        if not packet.game_info.is_round_active:
            blue, orange = total_goals(packet)
            if self.blue is None or self.orange is None:
                self.blue, self.orange = blue, orange
            if blue != self.blue:
                self.train_step(0)
                self.blue = blue
            if orange != self.orange:
                self.train_step(1)
                self.orange = orange
            return self.empty_controller
        if packet.game_cars[self.index].is_demolished:
            return self.empty_controller

        arr = self.input_formatter.create_input_array([packet], batch_size=1)
        arr = [self.torch.from_numpy(x).float() for x in arr]
        assert (arr[0].size() == (1, 3, 9))
        assert (arr[1].size() == (1, 5))
        # print(arr[0])

        mask = self.output_formatter.get_mask(packet)
        mask = self.torch.from_numpy(mask).byte()
        assert (mask.size() == (1, 13))

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data *= 0.99

        output, _ = self.model.forward(*arr)

        self.offset *= self.theta
        new_output = self.torch.distributions.Normal(output + self.offset, self.sigma).sample().clamp(-1, 1)
        assert (new_output.size() == (1, 13))

        self.offset = (new_output - output).detach()

        dist = self.torch.distributions.Normal(output, self.sigma / (1 - self.theta), validate_args=True)
        log_prob = dist.log_prob(new_output)
        log_prob[mask].sum().neg().backward()

        game_info_state = GameInfoState(game_speed=3.0)
        game_state = GameState(game_info=game_info_state)
        self.set_game_state(game_state)

        return self.output_formatter.format_controller_output(new_output[0], packet)

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    def load_config(self, config_object_header: ConfigHeader):
        self.model_path = config_object_header.get('model_path')
        self.load_model = config_object_header.getboolean('load_model')

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('model_path', str, default=os.path.join('models', 'cool_atba.mdl'),
                         description='Path to the model file')
        params.add_value('load_model', bool, default=False, description='The model should be loaded')

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = self.get_manager_path()
        key = 'swarm_manager'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        request.model_path = self.model_path
        request.load_model = self.load_model
        return request

    def train_step(self, winning_team: int):
        self.score *= 0.9
        self.score += self.team == winning_team

        if winning_team != self.team:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data *= -1

        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         p.grad.data /= self.steps
        # self.steps = 0

        self.optimizer.step()
        self.optimizer.zero_grad()
        acc_delta = []
        niter = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                niter = max(state['step'], niter)
                acc_delta.append(state['acc_delta'].mean())

        avg = self.torch.stack(acc_delta).mean().sqrt()

        self.writer.add_scalar('acc_delta', avg.item(), niter)
        self.writer.add_scalar('score', self.score, niter)

        file_path = os.path.join(get_repo_directory(), f'models/opt{self.team}.sd')
        self.torch.save(self.optimizer.state_dict(), file_path)

        if self.team:
            file_path = os.path.join(get_repo_directory(), f'models/snap{niter}.mdl')
            self.torch.save(self.model.state_dict(), file_path)

    def retire(self):
        print('yay')


def total_goals(packet):
    blue = 0
    orange = 0
    for car in range(packet.num_cars):
        if packet.game_cars[car].team == 0:
            blue += packet.game_cars[car].score_info.goals
        else:
            orange += packet.game_cars[car].score_info.goals

    return blue, orange
