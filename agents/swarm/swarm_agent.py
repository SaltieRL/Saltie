from rlbot.botmanager.helper_process_request import HelperProcessRequest
from agents.main_agent.base_model_agent import BaseModelAgent


class SwarmAgent(BaseModelAgent):

    pipe = None
    model = None
    input_formatter = None
    output_formatter = None
    game_memory = None

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = self.get_manager_path()
        key = 'swarm_manager'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request

    def get_manager_path(self):
        raise NotImplementedError()

    def initialize_agent(self):
        self.model = self.pipe.recv()
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()
        self.game_memory = self.pipe.recv()

    def create_model(self):
        return None
