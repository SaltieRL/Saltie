import os

from conversions.input.input_formatter import get_state_dim
from modelHelpers import action_handler
from modelHelpers import reward_manager
import time

import tensorflow as tf


class RewardTrainer:
    model_class = None
    learning_rate = 0.3

    file_number = 0

    epoch = 0
    display_step = 5

    last_action = None
    reward_manager = None
    local_pair_number = 0
    last_pair_number = 0
    train_time_difference = 0
    action_time_difference = 0

    def __init__(self):
        #config = tf.ConfigProto(
        #    device_count={'GPU': 1}
        #)
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        # writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))

        self.action_handler = action_handler.ActionHandler(split_mode=True)

        self.state_dim = get_state_dim()
        print('state size ' + str(self.state_dim))
        self.num_actions = self.action_handler.get_action_size()
        self.agent = self.get_model()(self.sess, self.state_dim, self.num_actions, action_handler=self.action_handler, is_training=True)

        self.agent.summary_writer = tf.summary.FileWriter(
            'training/events/{}-experiment'.format(self.agent.get_model_name()))

        self.agent.create_model()

        self.agent.create_reinforcement_training_model()

        self.agent.initialize_model()

        self.batch_size = self.agent.batch_size


    def get_model(self):
        #return rnn_atba.RNNAtba
        #return nnatba.NNAtba
        #return base_actor_critic.BaseActorCritic
        return self.model_class

    def start_new_file(self):
        self.file_number += 1
        self.last_action = None
        self.reward_manager = reward_manager.RewardManager()

    def process_pair(self, input_array, output_array, pair_number, file_version):
        # extra_features = feature_creator.get_extra_features_from_array(input_array)

        if self.last_action is not None:
            reward = 0  # self.reward_manager.get_reward(input_array)
            self.agent.store_rollout(input_state=input_array, last_action=self.last_action, reward=reward)

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
                    output_array[i * self.batch_size: (i + 1) * self.batch_size])
                counter = i
            if counter * self.batch_size < len(input_array):
                self.process_pair_batch(
                    input_array[counter * self.batch_size:],
                    output_array[counter * self.batch_size:])
            return

        self.agent.store_rollout(input_state=input_array, last_action=output_array, reward=[])
        #self.agent.store_rollout_batch(input_state=input_array, last_action=output_array)
        self.local_pair_number += (pair_number - self.last_pair_number)
        self.last_pair_number = pair_number

        if self.local_pair_number >= self.batch_size:
            self.local_pair_number = 0
            self.batch_process()


    def batch_process(self):
        start = time.time()
        self.agent.update_model()
        self.train_time_difference += time.time() - start
        # Display logs per step
    #    if self.epoch % self.display_step == 0:
    #        print("File:", '%04d' % self.file_number, "Epoch:", '%04d' % (self.epoch+1))
        self.epoch += 1

    def end_file(self):
        self.batch_process()
        print('\naction conversion time', self.action_time_difference)
        print('training time', self.train_time_difference)
        self.action_time_difference = 0
        self.train_time_difference = 0
        if self.file_number % 100 == 0:
            file_path = self.agent.get_model_path(self.agent.get_default_file_name() + str(self.file_number) + ".ckpt")
            self.save_replay(file_path)

    def end_everything(self):
        file_path = self.agent.get_model_path(self.agent.get_default_file_name() + ".ckpt")
        self.save_replay(file_path)

    def save_replay(self, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.agent.saver.save(self.sess, file_path)

