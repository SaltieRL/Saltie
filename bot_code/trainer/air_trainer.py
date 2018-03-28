import numpy as np
import tensorflow as tf

from bot_code.trainer.base_classes.default_model_trainer import DefaultModelTrainer
from bot_code.trainer.base_classes.download_trainer import DownloadTrainer
from bot_code.trainer.utils import controller_statistics
# from bot_code.modelHelpers.actions import action_factory
from bot_code.conversions.input.tensorflow_input_formatter import TensorflowInputFormatter

from collections import deque


class AirTrainer(DownloadTrainer):
    file_number = 0
    should_batch_process = True
    frames_per_input = 5

    learning_rate = 0.1
    input_dim = None
    output_dim = None


    def get_config_name(self):
        return 'air_trainer.cfg'

    def get_event_filename(self):
        return 'air_trainer'

    def load_config(self):
        super().load_config()
        config = super().create_config()
        try:
            self.learning_rate = config.getfloat(self.OPTIMIZER_CONFIG_HEADER, 'learning_rate')
        except Exception as e:
            self.learning_rate = 0.001

        try:
            self.input_dim = config.getfloat(self.MODEL_CONFIG_HEADER, 'input_dim')
        except Exception as e:
            self.input_dim = 219
        try:
            self.output_dim = config.getfloat(self.MODEL_CONFIG_HEADER, 'output_dim')
        except Exception as e:
            self.output_dim = 5

    def setup_trainer(self):
        super().setup_trainer()

    def setup_model(self):
        super().setup_model()
        self.model.create_model()
        # self.model.create_reinforcement_training_model()
        self.model.create_savers()
        self.model.initialize_model()


    def setup_trainer(self):
        DownloadTrainer.setup_trainer(self)
        session_config = tf.ConfigProto()
        self.sess = tf.Session(config=session_config)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.input_formatter = TensorflowInputFormatter(0, 0, self.batch_size, None)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def instantiate_model(self, model_class):
        return model_class(self.sess,
                           self.input_dim,
                           self.output_dim,
                           is_training=True,
                           optimizer=self.optimizer,
                           config_file=self.create_model_config())

    def start_new_file(self):

        self.input_batch = []
        self.label_batch = []
        self.input_game_tick = []

        self.file_number += 1

    def add_pair(self, input_array, output_array):
        self.input_batch.append(input_array)


        label = output_array

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

        input_batch = np.array(self.input_batch)
        # input_batch = self.model.input_formatter.format_array(input_batch)

        output = np.argwhere(np.isnan(input_batch))
        if len(output) > 0:
            print('nan indexes', output)
            for index in output:
                input_batch[index[0]][index[1]] = 0

        self.label_batch = np.array(self.label_batch, dtype=np.float32)

        feed_dict = self.model.create_feed_dict(input_batch, self.label_batch)
        self.model.run_train_step(True, feed_dict=feed_dict)

        self.epoch += 1

    def end_file(self):
        self.batch_process()
        if self.file_number % 100 == 0:
            self.model.save_model(model_path=None, global_step=self.file_number, quick_save=True)

    def end_everything(self):
        self.model.save_model()


if __name__ == '__main__':
    AirTrainer().run()
