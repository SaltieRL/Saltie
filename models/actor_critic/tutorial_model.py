from models import base_model
from models.actor_critic.policy_gradient import PolicyGradient
import tensorflow as tf
import numpy as np


class TutorialModel(PolicyGradient):
    max_gradient = 10.0
    total_loss_divider = 2.0
    # hidden_layer_activation = tf.nn.relu6
    # hidden_layer_activation = tf.tanh

    def __init__(self, session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None,
                 teacher=None
                 ):
        if teacher is not None:
            self.teacher = '_' + teacher
        else:
            self.teacher = ''
        super().__init__(session, num_actions,
                         input_formatter_info=input_formatter_info,
                         player_index=player_index,
                         action_handler=action_handler,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

    def printParameters(self):
        super().printParameters()
        print('TutorialModel Parameters:')
        print('Teacher:', self.teacher)

    def load_config_file(self):
        super().load_config_file()

        try:
            self.teacher = '_' + self.config_file.get(base_model.MODEL_CONFIGURATION_HEADER,
                                                             'teacher')
        except:
            print('unable to load the teacher')

        self.num_split_layers = min(self.num_split_layers, self.num_layers - 2)

    def create_training_op(self, logprobs, labels):
        actor_gradients, actor_loss, actor_reg_loss = self.create_actor_gradients(logprobs, labels)

        tf.summary.scalar("total_reg_loss", actor_reg_loss)

        return self._compute_training_op(actor_gradients, [])

    def create_advantages(self):
        return tf.constant(1.0)

    def fancy_calculate_number_of_ones(self, number):
        """Only use this once it is supported"""
        # https://blogs.msdn.microsoft.com/jeuge/2005/06/08/bit-fiddling-3/
        # uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
        # return ((uCount + (uCount >> 3)) & 030707070707) % 63;

        threes_64 = tf.constant(0o033333333333, dtype=tf.int64)
        full_ones = tf.constant(0o011111111111, dtype=tf.int64)
        sevens = tf.constant(0o030707070707, dtype=tf.int64)

        bitwise1 = tf.bitwise.bitwise_and(number // 2, threes_64)
        bitwise2 = tf.bitwise.bitwise_and(number // 4, full_ones)
        uCount = number - bitwise1 - bitwise2

        bitwise3 = tf.bitwise.bitwise_and(uCount + (uCount // 8), sevens)
        return tf.mod(bitwise3, 63)

    def calculate_loss_of_actor(self, logprobs, taken_actions, index):
        cross_entropy_loss, initial_wrongness, __ = super().calculate_loss_of_actor(logprobs, taken_actions, index)
        wrongness = tf.constant(initial_wrongness)
        argmax = tf.argmax(logprobs, axis=1)
        if self.action_handler.action_list_names[index] != 'combo':
            if self.action_handler.is_classification(index):
                wrongness += tf.cast(tf.abs(tf.cast(argmax, tf.float32) - taken_actions), tf.float32)
            else:
                wrongness += tf.abs(taken_actions - tf.round(logprobs * 2.0) / 2.0)
        else:
            # use temporarily
            # wrongness += tf.log(1.0 + tf.cast(tf.abs(tf.cast(argmax, tf.float32) - taken_actions), tf.float32))

            number = tf.bitwise.bitwise_xor(tf.cast(self.argmax[index], tf.int64), tf.cast(taken_actions, tf.int64))
            wrongness += tf.cast(self.fancy_calculate_number_of_ones(number), tf.float32)

        return cross_entropy_loss, wrongness, False

    def get_model_name(self):
        return 'tutorial_bot' + ('_split' if self.action_handler.is_split_mode else '') + self.teacher

    def add_histograms(self, gradients):
        # summarize gradients
        for grad, var in gradients:
            # we do not need to see variables
            # tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + '/gradients', grad)
