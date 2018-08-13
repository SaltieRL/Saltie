import psutil
from rlbot.botmanager.bot_helper_process import BotHelperProcess
from rlbot.utils.logging_utils import get_logger

from framework.model_holder.base_model_holder import BaseModelHolder

from multiprocessing.managers import BaseManager
from swarm_trainer.reward_memory import BaseRewardMemory


class BaseHiveManager(BotHelperProcess):

    batch_size = 1000

    def __init__(self, agent_metadata_queue, quit_event):
        super().__init__(agent_metadata_queue, quit_event)
        self.logger = get_logger('base_hive_mgr')
        self.metadata_map = []

        self.manager = self.setup_manager()
        self.game_memory = self.manager.Memory()
        self.actor_model = self.get_model()

    def get_model(self) -> BaseModelHolder:
        pass

    def setup_manager(self):
        BaseManager.register('Memory', BaseRewardMemory)
        manager = BaseManager()
        manager.start()

        return manager

    def get_shared_model_handle(self):
        return "hello"

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe

            pipe.send(self.get_shared_model_handle())
            pipe.send(self.game_memory)

            self.metadata_map[metadata.team] = metadata

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
        while not self.quit_event.is_set():
            self.learn_memory()

        # quit -> save actor network

        self.logger.info('saving model')
        self.actor_model.finish_training()
        self.logger.info('model saved')
        self.game_memory.save(self.actor_model.get_model_name() + '.exp')

    def learn_memory(self):
        input_data, action, reward = self.game_memory.get_sample(self.batch_size)
        if len(input_data) > 0:
            if len(reward) == 0:
                reward = None
            self.actor_model.train_step(input_array=input_data, output_array=action,
                                        rewards=reward, batch_size=len(input_data))
