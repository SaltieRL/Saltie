from rlbot.botmanager.helper_process_request import HelperProcessRequest
from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.logging_utils import get_logger
from framework.utils import get_repo_directory
import sys


class SwarmAgent(BaseAgent):

    pipe = None
    model = None
    input_formatter = None
    output_formatter = None
    game_memory = None

    optimizer = None
    loss_function = None

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.logger = get_logger(name)
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = self.get_manager_path()
        key = 'swarm_manager'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request

    def get_manager_path(self):
        raise NotImplementedError

    def create_input_formatter(self):
        raise NotImplementedError

    def create_output_formatter(self):
        raise NotImplementedError

    def initialize_agent(self):
        self.model = self.pipe.recv()
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()
        self.game_memory = self.pipe.recv()

        self.optimizer = self.torch.optim.Adamax(self.model.parameters())
        self.loss_function = self.torch.nn.L1Loss()
