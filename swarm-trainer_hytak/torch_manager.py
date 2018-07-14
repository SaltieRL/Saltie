from rlbot.botmanager.bot_helper_process import BotHelperProcess
from rlbot.utils.logging_utils import get_logger
import sys
from time import sleep
import psutil
from tensorboardX import SummaryWriter

name = 'online'
load_model = False
load_exp = False


class TorchManager(BotHelperProcess):
    def __init__(self, agent_metadata_queue, quit_event):
        super().__init__(agent_metadata_queue, quit_event)

        from multiprocessing.managers import BaseManager
        from leviathan.torch_memory import RewardMemory
        BaseManager.register('Memory', RewardMemory)

        self.logger = get_logger('torch_mgr')
        self.metadata_list = [None, None]
        self.manager = BaseManager()
        self.manager.start()
        self.game_memory = self.manager.Memory()
        self.writer = SummaryWriter()
        self.n_iter = 0

        if load_exp:
            self.game_memory.load(name + '.exp')

        from leviathan.torch_model import SymmetricModel, RewardModel
        self.actor_model = SymmetricModel()

        import torch
        if load_model:
            self.actor_model.load_state_dict(torch.load(name + '.actor'))

        self.actor_model.share_memory()

        self.reward_model = RewardModel(self.actor_model)
        self.optimizer = torch.optim.Adamax(self.reward_model.parameters())
        self.loss_function = torch.nn.L1Loss()

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe

            pipe.send(self.actor_model)
            pipe.send(self.game_memory)

            self.metadata_list[metadata.team] = metadata

        self.logger.info('set up all agents')

        my_process = psutil.Process()
        my_process.cpu_affinity([1, 2, 3])
        my_process.nice(psutil.HIGH_PRIORITY_CLASS)

        self.game_loop()

    def game_loop(self):
        while not self.quit_event.is_set():
            self.learn_memory()

        # quit -> save actor network

        import torch
        self.logger.info('saving model')
        torch.save(self.actor_model.state_dict(), name + '.actor')
        self.logger.info('model saved')
        self.game_memory.save(name + '.exp')

    def learn_memory(self):
        spatial, car_stats, action, reward = self.game_memory.get_sample(1000)
        if len(reward) > 0:
            self.optimizer.zero_grad()

            estimated_reward = self.reward_model.forward(spatial, car_stats, action)

            loss = self.loss_function(estimated_reward, reward)

            self.writer.add_scalar('loss', loss, self.n_iter)
            self.n_iter += 1

            loss.backward()
            self.optimizer.step()
