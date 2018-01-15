import numpy as np

from conversions.input.input_formatter import get_state_dim
from trainer.base_classes.default_model_trainer import DefaultModelTrainer
from trainer.base_classes.download_trainer import DownloadTrainer
from trainer.utils import controller_statistics
from trainer.utils.trainer_runner import run_trainer


class CopyTrainer(DownloadTrainer, DefaultModelTrainer):

    should_shuffle = False
    file_number = 0

    epoch = 0
    display_step = 5

    batch_size = 1000
    input_game_tick = []
    input_batch = []
    label_batch = []
    eval_file = False
    eval_number = 30
    controller_stats = None
    action_length = None

    def load_config(self):
        super().load_config()
        config = super().create_config()
        try:
            self.should_shuffle = config.getboolean(self.DOWNLOAD_TRAINER_CONFIGURATION_HEADER,
                                                   'download_files')
        except Exception as e:
            self.should_shuffle = True

    def get_config_name(self):
        return 'copy_trainer.cfg'

    def get_event_filename(self):
        return 'copy_replays'

    def instantiate_model(self, model_class):
        return model_class(self.sess,
                           self.action_handler.get_logit_size(), action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_config(), teacher='replay_files')

    def setup_model(self):
        super().setup_model()
        self.model.create_model()
        self.model.create_copy_training_model()
        self.model.create_savers()
        self.model.initialize_model()
        self.controller_stats = controller_statistics.OutputChecks(self.sess, self.action_handler,
                                                                   self.batch_size, self.model.smart_max,
                                                                   model_placeholder=self.model.input_placeholder)
        self.controller_stats.create_model()

    def start_new_file(self):

        self.input_batch = []
        self.label_batch = []
        self.input_game_tick = []
        if self.file_number % self.eval_number == 0:
            self.eval_file = True
            self.action_length = self.action_handler.control_size
        else:
            self.eval_file = False
            self.action_length = self.action_handler.get_number_actions()
        self.file_number += 1

    def add_pair(self, input_array, output_array):
        self.input_batch.append(input_array)

        if self.eval_file:
            label = output_array
        else:
            label = self.action_handler.create_action_index(output_array)
        # print(output_array)
        # print(label)
        self.label_batch.append(label)

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def process_pair(self, input_array, output_array, pair_number, file_version):
        self.add_pair(input_array, output_array)
        if len(self.input_batch) == self.batch_size:
            self.batch_process()
            self.input_batch = []
            self.label_batch = []
            self.input_game_tick = []
            # do stuff

    def batch_process(self):
        if len(self.input_batch) <= 1 or len(self.label_batch) <= 1:
            return

        input_length = len(self.input_batch)
        self.input_batch = np.array(self.input_batch)
        self.input_batch = self.input_batch.reshape(input_length, get_state_dim())

        output = np.argwhere(np.isnan(self.input_batch))
        if len(output) > 0:
            print('nan indexes', output)
            for index in output:
                self.input_batch[index[0]][index[1]] = 0

        self.label_batch = np.array(self.label_batch)
        self.label_batch = self.label_batch.reshape(input_length, self.action_length)

        print(input_length)
        if self.should_shuffle:
            self.input_batch, self.label_batch = self.unison_shuffled_copies(self.input_batch, self.label_batch)

        if self.eval_file:
            self.controller_stats.get_amounts(input_array=self.input_batch, bot_output=np.transpose(self.label_batch))
        else:
            self.model.run_train_step(True, {self.model.get_input_placeholder(): self.input_batch,
                                             self.model.get_labels_placeholder(): self.label_batch})

        self.epoch += 1

    def end_file(self):
        self.batch_process()
        if self.file_number % 100 == 0:
            self.model.save_model(model_path=None, global_step=self.file_number, quick_save=True)

    def end_everything(self):
        self.model.save_model()


if __name__ == '__main__':
    run_trainer(trainer=CopyTrainer())
