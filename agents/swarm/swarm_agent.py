import os

from rlbot.botmanager.helper_process_request import HelperProcessRequest
from rlbot.agents.base_agent import BaseAgent, BOT_CONFIG_AGENT_HEADER
from rlbot.parsing.custom_config import ConfigHeader, ConfigObject
from rlbot.utils.logging_utils import get_logger
from framework.utils import get_repo_directory
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        self.manager_path = None
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = self.get_manager_path()
        key = 'swarm_manager'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request

    def load_config(self, config_object_header: ConfigHeader):
        self.manager_path = config_object_header.get_string('manager_path')

    def get_manager_path(self):
        return os.path.join(path, self.manager_path)

    def create_input_formatter(self):
        raise NotImplementedError

    def create_output_formatter(self):
        raise NotImplementedError

    def initialize_agent(self):
        self.model = self.pipe.recv()
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()
        self.game_memory = self.pipe.recv()

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        super(SwarmAgent, SwarmAgent).create_agent_configurations(config)
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('manager_path', str, default=os.path.join('examples', 'levi', 'torch_manager'),
                         description='Path to the manager bot')
