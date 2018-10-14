import sys
import psutil
from rlbot.botmanager.bot_helper_process import BotHelperProcess
from rlbot.utils.logging_utils import get_logger
import time

from framework.utils import get_repo_directory


class BaseHiveManager(BotHelperProcess):

    batch_size = 2000

    def __init__(self, agent_metadata_queue, quit_event):
        super().__init__(agent_metadata_queue, quit_event)
        self.logger = get_logger('base_hive_mgr')
        sys.path.insert(0, get_repo_directory())  # this is for separate process imports

        self.manager = self.setup_manager()
        self.game_memory = self.manager.Memory()
        self.actor_model = self.get_model()
        self.shared_model_handle = self.get_shared_model_handle()

        self.setup_trainer()

    def get_model(self):
        raise NotImplementedError()

    def get_shared_model_handle(self):
        raise NotImplementedError()

    @staticmethod
    def setup_manager():
        from multiprocessing.managers import BaseManager
        from swarm_trainer.reward_memory import BaseRewardMemory

        BaseManager.register('Memory', BaseRewardMemory)
        manager = BaseManager()
        manager.start()

        return manager

    def setup_trainer(self):
        raise NotImplementedError()

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe

            pipe.send(self.shared_model_handle)
            pipe.send(self.game_memory)

        self.logger.info('set up all agents')

        my_process = psutil.Process()
        my_process.cpu_affinity([1, 2, 3])
        my_process.nice(psutil.HIGH_PRIORITY_CLASS)

        self.game_loop()

    def game_loop(self):
        """
        Loops through the game providing training as data is collected.
        :return:
        """
        self.initialize_training()

        while not self.quit_event.is_set():
            self.learn_memory()

        self.finish_training()

    def learn_memory(self):
        input_data, action, reward = self.game_memory.get_random_sample(self.batch_size)
        if action.shape[0] >= 1000:
            self.train_step(formatted_input=input_data, formatted_output=action,
                            rewards=reward, batch_size=action.shape[0])
        else:
            time.sleep(5)

    def initialize_training(self, load_model=False, load_exp=False):
        raise NotImplementedError()

    def train_step(self, formatted_input, formatted_output, rewards=None, batch_size=1):
        """
        Performs a single train step on the data given.
        All data (input, output, rewards) should end up producing arrays of the same length
        :param formatted_input: Fed as input to the model this is the data that is expected to produce results.
        :param formatted_output: The expected result that the model should produce.
        :param rewards: Optional, rewards are weighted values to say how strongly a certain action should be copied.
        :param batch_size: How many are in the array
        :return:
        """
        raise NotImplementedError()

    def finish_training(self, save_model=True, save_exp=False):
        raise NotImplementedError()

    def get_model_name(self):
        return str(type(self.actor_model).__name__)

    def get_file_path(self):
        return get_repo_directory() + '/trainer/weights/' + self.get_model_name() + '.mdl'
