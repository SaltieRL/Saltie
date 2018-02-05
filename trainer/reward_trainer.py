from modelHelpers import reward_manager
import time

from trainer.base_classes.default_model_trainer import DefaultModelTrainer
from trainer.base_classes.download_trainer import DownloadTrainer
from trainer.utils.trainer_runner import run_trainer


class RewardTrainer(DownloadTrainer, DefaultModelTrainer):
    file_number = 0

    epoch = 0
    display_step = 5

    last_action = None
    reward_manager = None
    local_pair_number = 0
    last_pair_number = 0
    train_time_difference = 0
    action_time_difference = 0

    def load_config(self):
        super().load_config()

    def setup_trainer(self):
        super().setup_trainer()

    def get_config_name(self):
        return 'reward_trainer.cfg'

    def setup_model(self):
        super().setup_model()
        self.model.create_model()
        self.model.create_reinforcement_training_model()
        self.model.create_savers()
        self.model.initialize_model()

    def start_new_file(self):
        self.file_number += 1
        self.last_action = None
        self.reward_manager = reward_manager.RewardManager()

    def process_pair(self, input_array, output_array, pair_number, file_version):
        if self.last_action is not None:
            reward = 0  # self.reward_manager.get_reward(input_array)
            self.model.store_rollout(input_state=input_array, last_action=self.last_action, reward=reward)

        start = time.time()
        self.last_action = self.action_handler.create_action_index(output_array)
        self.action_time_difference += time.time() - start

        if pair_number % self.batch_size == 0 and pair_number != 0:
            self.batch_process()

    def process_pair_batch(self, input_array, output_array, pair_number, file_version):
        # extra_features = feature_creator.get_extra_features_from_array(input_array)

        if len(input_array) > self.batch_size:
            print('splitting up!')
            counter = 0
            for i in range(int(len(input_array) / self.batch_size)):
                self.process_pair_batch(
                    input_array[i * self.batch_size: (i + 1) * self.batch_size],
                    output_array[i * self.batch_size: (i + 1) * self.batch_size], pair_number + self.batch_size,
                    file_version)
                counter = i
            if counter * self.batch_size < len(input_array):
                batch_number = len(input_array) - (counter * self.batch_size)
                self.process_pair_batch(
                    input_array[counter * self.batch_size:],
                    output_array[counter * self.batch_size:], file_version, pair_number + batch_number)
                self.local_pair_number = batch_number
            self.last_pair_number = pair_number
        else:
            self.model.store_rollout(input_state=input_array, last_action=output_array, reward=[])
            self.local_pair_number += (pair_number - self.last_pair_number)
            self.last_pair_number = pair_number

            if self.local_pair_number >= self.batch_size:
                self.local_pair_number = 0
                self.batch_process()

    def batch_process(self):
        start = time.time()
        self.model.update_model()
        self.train_time_difference += time.time() - start
        # Display logs per step
        # if self.epoch % self.display_step == 0:
        #     print("File:", '%04d' % self.file_number, "Epoch:", '%04d' % (self.epoch+1))
        self.epoch += 1

    def end_file(self):
        self.batch_process()
        print('\naction conversion time', self.action_time_difference)
        print('training time', self.train_time_difference)
        self.action_time_difference = 0
        self.train_time_difference = 0
        if self.file_number % 100 == 0:
            self.model.save_model(model_path=None,
                                  global_step=self.file_number, quick_save=True)

    def end_everything(self):
        self.model.save_model()


if __name__ == '__main__':
    run_trainer(trainer=RewardTrainer())
