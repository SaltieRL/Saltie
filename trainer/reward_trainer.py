from conversions import output_formatter
from conversions.input_formatter import get_state_dim_with_features, InputFormatter
from modelHelpers import action_handler
from modelHelpers import feature_creator
from modelHelpers import reward_manager
from models.atbas import rnn_atba
from models.actor_critic import base_actor_critic
from models.actor_critic import policy_gradient
from models.atbas import nnatba

import numpy as np
import tensorflow as tf


class RewardTrainer:
    model_class = None
    learning_rate = 0.3

    file_number = 0

    epoch = 0
    display_step = 5

    batch_size = 2000
    last_action = None
    reward_manager = None

    def __init__(self):
        #config = tf.ConfigProto(
        #    device_count={'GPU': 1}
        #)
        #self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        # writer = tf.summary.FileWriter('tmp/{}-experiment'.format(random.randint(0, 1000000)))

        self.action_handler = action_handler.ActionHandler(split_mode=True)

        self.state_dim = get_state_dim_with_features()
        print('state size ' + str(self.state_dim))
        self.num_actions = self.action_handler.get_action_size()
        self.agent = self.get_model()(self.sess, self.state_dim, self.num_actions, self.action_handler, is_training=True)

        self.agent.initialize_model()

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
        extra_features = feature_creator.get_extra_features_from_array(input_array)

        input_state = np.append(input_array, extra_features)

        if self.last_action is not None:
            reward = self.reward_manager.get_reward(input_state)
            self.agent.store_rollout(input_state=input_state, last_action=self.last_action, reward=reward)

        self.last_action = self.action_handler.create_action_index(output_array)

        if pair_number % self.batch_size == 0 and pair_number != 0:
            self.batch_process()


    def batch_process(self):
        self.agent.update_model()
        # Display logs per step
        if self.epoch % self.display_step == 0:
            print("File:", '%04d' % self.file_number, "Epoch:", '%04d' % (self.epoch+1))
        self.epoch += 1

    def end_file(self):
        self.batch_process()
        if self.file_number % 100 == 0:
            saver = tf.train.Saver()
            file_path = self.agent.get_model_path(self.agent.get_default_file_name() + str(self.file_number) + ".ckpt")
            saver.save(self.sess, file_path)

    def end_everything(self):
        saver = tf.train.Saver()
        file_path = self.agent.get_model_path(self.agent.get_default_file_name() + ".ckpt")
        saver.save(self.sess, file_path)
