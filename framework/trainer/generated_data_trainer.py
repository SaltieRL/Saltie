from framework.data_generator.base_generator import BaseDataGenerator
from framework.model_holder.base_model_holder import BaseModelHolder
from framework.trainer.base_trainer import BaseTrainer


class GeneratedDataTrainer(BaseTrainer):

    def __init__(self, model_holder: BaseModelHolder, data_generator: BaseDataGenerator):
        super().__init__(model_holder)
        self.data_generator = data_generator

    def initialize_training(self, **kwargs):
        self.data_generator.initialize(**kwargs)

    def train(self):
        for data in self.data_generator.get_data():
            self.model_holder.train_step(data, data)
