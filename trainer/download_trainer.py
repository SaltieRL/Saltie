from model_holder.base_model_holder import BaseModelHolder
from trainer.base_trainer import BaseTrainer
from trainer.downloader import Downloader


class DownloadTrainer(BaseTrainer):
    def __init__(self, model_holder: BaseModelHolder):
        super().__init__(model_holder)
        self.downloader = Downloader()

