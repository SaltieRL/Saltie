import os

from rlbot.botmanager.helper_process_request import HelperProcessRequest
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent, BOT_CONFIG_AGENT_HEADER
from rlbot.parsing.custom_config import ConfigHeader, ConfigObject
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
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports
        self.logger = get_logger(name)
        self.manager_path = None
        self.model_path = None
        self.load_model = None
        self.empty_controller = SimpleControllerState()

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = self.get_manager_path()
        key = 'swarm_manager'
        options = {'model_path': self.model_path, 'load_model': self.load_model}
        request = HelperProcessRequest(file, key, options=options)
        self.pipe, request.pipe = Pipe(False)
        return request

    def load_config(self, config_object_header: ConfigHeader):
        self.model_path = config_object_header.get('model_path')
        self.load_model = config_object_header.getboolean('load_model')

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

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        super(SwarmAgent, SwarmAgent).create_agent_configurations(config)
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('model_path', str, default=os.path.join('models', 'cool_atba.mdl'),
                         description='Path to the model file')
        params.add_value('load_model', bool, default=False, description='The model should be loaded')
