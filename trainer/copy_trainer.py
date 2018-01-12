from conversions.input import tensorflow_input_formatter
from conversions.input.input_formatter import get_state_dim
from modelHelpers.actions import action_handler, action_factory, dynamic_action_handler
from modelHelpers import feature_creator
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from models.actor_critic import base_actor_critic

import numpy as np
import tensorflow as tf

from trainer.base_classes.default_model_trainer import DefaultModelTrainer
from trainer.base_classes.download_trainer import DownloadTrainer
from trainer.utils.trainer_runner import run_trainer


class CopyTrainer(DownloadTrainer, DefaultModelTrainer):

    file_number = 0

    epoch = 0
    display_step = 5

    batch_size = 1000
    input_game_tick = []
    input_batch = []
    label_batch = []

    def instantiate_model(self, model_class):
        return model_class(self.sess, self.input_formatter.get_state_dim_with_features(),
                           self.action_handler.get_logit_size(), action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_config(), teacher='replay_files')

    def setup_model(self):
        super().setup_model()
        self.model.create_model()
        self.model.create_copy_training_model()
        self.model.initialize_model()

    def start_new_file(self):
        self.file_number += 1
        self.input_batch = []
        self.label_batch = []
        self.input_game_tick = []

    def add_pair(self, input_array, output_array):
        self.input_batch.append(input_array)

        label = self.action_handler.create_action_index(output_array)
        self.label_batch.append(label)

    def process_pair(self, input_array, output_array, pair_number, file_version):
        self.add_pair(input_array, output_array)
        if len(self.input_batch) == self.batch_size:
            self.batch_process()
            self.input_batch = []
            self.label_batch = []
            self.input_game_tick = []
            # do stuff

    def batch_process(self):
        if len(self.input_batch) == 0 or len(self.label_batch) == 0:
            print('batch was empty quitting')
            return

        self.input_batch = np.array(self.input_batch)
        self.input_batch = self.input_batch.reshape(self.batch_size, self.input_formatter.get_state_dim_with_features())

        self.label_batch = np.array(self.label_batch)
        self.label_batch = self.label_batch.reshape(self.batch_size, self.input_formatter.get_state_dim_with_features())

        self.model.run_train_step(True, self.input_batch, self.label_batch)

        # Display logs per step
        if self.epoch % self.display_step == 0:
            pass
        self.epoch += 1

    def end_file(self):
        self.batch_process()
        if self.file_number % 100 == 0:
            self.model.save_model(model_path=None, global_step=self.file_number, quick_save=True)

    def end_everything(self):
        self.model.save_model()


if __name__ == '__main__':
    run_trainer(trainer=CopyTrainer())
