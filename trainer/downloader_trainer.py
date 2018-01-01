import os

import time
import numpy as np
from models.downloader_model import DownloaderModel


class DownloaderTrainer:
    model_class = DownloaderModel

    file_number = 0

    epoch = 0
    display_step = 5

    last_action = None
    local_pair_number = 0
    last_pair_number = 0
    train_time_difference = 0
    action_time_difference = 0

    def __init__(self, parse_data=True):
        self.parse_data = parse_data
        self.agent = self.get_model()(is_training=True)

        self.batch_size = self.agent.batch_size

    def get_model(self):
        return self.model_class

    def start_new_file(self):
        self.file_number += 1
        self.last_action = None

    def process_pair(self, input_array, output_array, pair_number, file_version):
        # extra_features = feature_creator.get_extra_features_from_array(input_array)

        self.agent.all_inputs.append(input_array)
        self.agent.all_outputs.append(output_array)
        # start = time.time()
        # self.action_time_difference += time.time() - start

        # print("In process_pair", len(self.agent.all_inputs))
        # print(pair_number, input_array.shape, output_array.shape, self.batch_size)

        if pair_number % self.batch_size == 0 and pair_number != 0:
            self.batch_process()

    def batch_process(self):
        start = time.time()
        # self.agent.update_model()
        self.train_time_difference += time.time() - start
        # Display logs per step
    #    if self.epoch % self.display_step == 0:
    #        print("File:", '%04d' % self.file_number, "Epoch:", '%04d' % (self.epoch+1))
        self.epoch += 1

    def end_file(self):
        # save numpy array inputs and outputs
        # self.batch_process()

        # print('\nAll inputs: %s, All outputs: %s' % (len(self.agent.all_inputs), len(self.agent.all_outputs)))

        self.agent.all_inputs = np.concatenate(self.agent.all_inputs, axis=0)
        self.agent.all_outputs = np.concatenate(self.agent.all_outputs, axis=0)

        if self.parse_data:
            self.parse_input_and_output_data()
        self.save_data()

        # train model on file
        # reset inputs and outputs
        self.agent.all_inputs = []
        self.agent.all_outputs = []

    def end_everything(self):
        print('Last file number: %s' % self.file_number)
        print('Finished saving data')

    def parse_input_and_output_data(self):
        """Parses data to only keep player position inputs in this frame and the next, parses output to extract pitch and roll. Used to generate data for a replay-labeller."""
        # data is stored in self.agent.all_inputs, self.agent.all_outputs

        # input section
        input_array = self.agent.all_inputs
        # keep only player's position, rotation, velocity, and angular velocity
        indices_to_keep = np.r_[9:21]
        input_array = input_array[:, indices_to_keep]
        # get every other row
        input_array = input_array[::2]
        # insert add frame x+1 input to frame x
        self.agent.all_inputs = np.append(
            input_array[:-1], input_array[1:], axis=1)
        # print(input_array.shape)

        # output section
        # get every other row
        output_array = self.agent.all_outputs[::2]
        # remove last row
        output_array = output_array[:-1]
        # get pitch and roll
        self.agent.all_outputs = output_array[:, [2, 4]]

    def save_data(self):
        print('\nAll inputs: %s, All outputs: %s' %
              (self.agent.all_inputs.shape, self.agent.all_outputs.shape))

        cwd = os.getcwd()

        os.makedirs(os.path.join(cwd, 'data'), exist_ok=True)

        np.save(os.path.join(cwd, 'data', 'inputs%s' %
                             self.file_number), self.agent.all_inputs)
        np.save(os.path.join(cwd, 'data', 'outputs%s' %
                             self.file_number), self.agent.all_outputs)

    def save_replay(self, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.agent.save(file_path)
