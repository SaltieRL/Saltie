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
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

import psutil
from rlbot.botmanager.bot_helper_process import BotHelperProcess
from rlbot.utils.logging_utils import get_logger
from quicktracer import trace
import numpy as np
from framework.utils import get_repo_directory
import multiprocessing

eps = np.finfo(np.float32).eps.item()


class LeviReinforcementManager(BotHelperProcess):
    optimizer = None
    distance = None
    threshold = None

    batch_size = 500
    memory_size = 100000

    def __init__(self, agent_metadata_queue, quit_event, options):
        super().__init__(agent_metadata_queue, quit_event, options)
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports
        self.logger = get_logger('base_hive_mgr')

        import torch
        self.torch = torch  # before initialization as this is needed for `setup_trainer`

        self.plot_count = 0

        self.actor_model = self.get_model()
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()

        self.saved_log_probs = {}
        self.team_map = {}
        self.pipes = []
        self.goal_number = 0

        self.model_path = None
        self.load_model = None

    @staticmethod
    def get_model():
        from examples.levi.torch_model import SymmetricModel
        return SymmetricModel().share_memory()

    def initialize_training(self, load_model=None):
        if load_model:
            file_path = self.get_file_path()
            self.actor_model.load_state_dict(self.torch.load(file_path))

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe
            self.model_path = metadata.helper_process_request.model_path
            self.load_model = metadata.helper_process_request.load_model

            pipe.send(self.actor_model)
            pipe.send(self.queue)

            self.team_map[metadata.index] = metadata.team
            self.saved_log_probs[metadata.index] = []
            self.pipes.append(pipe)

        self.logger.info('set up all agents')

        my_process = psutil.Process()
        my_process.cpu_affinity([0, 2])
        my_process.nice(psutil.NORMAL_PRIORITY_CLASS)

        self.game_loop()

    def game_loop(self):
        """
        Loops through the game providing training as data is collected.
        :return:
        """
        self.initialize_training(load_model=self.load_model)

        self.quit_event.wait()
        # while not self.quit_event.is_set():
        #     item = self.queue.get()
        #     if item['type'] == 'data':
        #         self.saved_log_probs[item['index']].append(item['data'])
        #     if item['type'] == 'goal' and item['goal_number'] > self.goal_number:
        #         self.goal_number = item['goal_number']
        #         self.train_step(item['winning_team'])

        self.finish_training()

    def finish_training(self, save_model=True):
        if save_model:
            file_path = self.get_file_path()
            print('saving model at:', file_path)
            self.torch.save(self.actor_model.state_dict(), file_path)

    def get_model_name(self):
        return str(type(self.actor_model).__name__)

    def get_file_path(self):
        return os.path.join(get_repo_directory(), self.model_path)
