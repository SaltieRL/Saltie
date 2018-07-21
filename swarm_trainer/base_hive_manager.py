import psutil
from rlbot.botmanager.bot_helper_process import BotHelperProcess
from rlbot.utils.logging_utils import get_logger

from framework.model_holder.base_model_holder import BaseModelHolder


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
        from multiprocessing.managers import BaseManager
        from swarm_trainer.reward_memory import RewardMemory

        BaseManager.register('Memory', RewardMemory)
        manager = BaseManager()
        manager.start()

        return manager

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe

            pipe.send(self.actor_model.model)
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
        spatial, car_stats, action, reward = self.game_memory.get_sample(self.batch_size)
        if len(reward) > 0:
            self.optimizer.zero_grad()

            estimated_reward = self.reward_model.forward(spatial, car_stats, action)

            loss = self.loss_function(estimated_reward, reward)

            self.writer.add_scalar('loss', loss, self.n_iter)
            self.n_iter += 1

            loss.backward()
            self.optimizer.step()
