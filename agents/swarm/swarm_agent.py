import os
from rlbot.botmanager.helper_process_request import HelperProcessRequest
from agents.online_training.online_training_agent import OnlineTrainingAgent


class SwarmAgent(OnlineTrainingAgent):

    pipe = None

    def get_helper_process_request(self) -> HelperProcessRequest:
        from multiprocessing import Pipe

        file = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hive_manager.py'))
        key = 'saltie_hive_mind'
        request = HelperProcessRequest(file, key)
        self.pipe, request.pipe = Pipe(False)
        return request
