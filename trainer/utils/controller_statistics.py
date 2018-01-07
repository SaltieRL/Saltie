from trainer.utils import random_packet_creator
from TutorialBot import tutorial_bot_output
from conversions.input import tensorflow_input_formatter
import tensorflow as tf
import numpy as np


class OutputChecks:
    model_output = None
    game_tick_packet = None
    accuracy_over_time = None
    bot_data_over_time = None

    def __init__(self, packets, model_output, game_tick_packet, input_array, tf_session, action_handler, tutorial_bot = None):
        self.sess = tf_session
        self.packets = packets
        self.game_tick_packet = game_tick_packet
        self.input_array = input_array
        self.packet_generator = random_packet_creator.TensorflowPacketGenerator(packets)
        self.tutorial_bot = tutorial_bot
        self.model_output = model_output
        self.actionHandler = action_handler

        if self.tutorial_bot is None:
            self.tutorial_bot = tutorial_bot_output.TutorialBotOutput(packets)

    def create_model(self):
        # clear history
        self.accuracy_over_time = []
        self.bot_data_over_time = []


    def get_amounts(self):
        controls = tf.transpose(
            self.actionHandler.create_tensorflow_controller_from_selection(self.model_output, self.packets))
        expected = self.tutorial_bot.get_output_vector(self.game_tick_packet)

        output, bot_output = self.sess.run([controls, expected])

        accuracy = np.sum(np.isclose(output, bot_output, 0.01), 1) / np.size(output[1])
        self.accuracy_over_time.append(accuracy)
        self.bot_data_over_time.append((output, bot_output))

        analog_buckets = [-1.0001, -0.50001, -0.0001, 0.0001, 0.50001, 1.0001]
        boolean_buckets = [-0.001, 0.50001, 1.0001]
        np.set_printoptions(formatter={'int': '{0:5}'.format})
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, -0>, [0], <0+, 0.5], <0.5, 1]")
        print("Real is model output, Expt is tutorialbot output and Acc. is accuracy")
        print("Throttle :  ")
        print("     Real:  ", np.histogram(output[0], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[0], analog_buckets)[0])
        print("     Acc.: ",  accuracy[0])
        print("Steer    :  ")
        print("     Real:  ", np.histogram(output[1], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[1], analog_buckets)[0])
        print("     Acc.: ",  accuracy[1])
        print("Pitch    :  ")
        print("     Real:  ", np.histogram(output[2], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[2], analog_buckets)[0])
        print("     Acc.: ",  accuracy[2])
        print("Yaw      :  ")
        print("     Real:  ", np.histogram(output[3], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[3], analog_buckets)[0])
        print("     Acc.: ",  accuracy[3])
        print("Roll     :  ")
        print("     Real:  ", np.histogram(output[4], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[4], analog_buckets)[0])
        print("     Acc.: ",  accuracy[4])
        print("From here the ranges are [0.0, 0.5>, [0.5, 1.0]")
        print("Jump     :  ")
        print("     Real:  ", np.histogram(output[5], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[5], boolean_buckets)[0])
        print("     Acc.: ",  accuracy[5])
        print("Boost    :  ")
        print("     Real:  ", np.histogram(output[6], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[6], boolean_buckets)[0])
        print("     Acc.: ",  accuracy[6])
        print("Handbrake:  ")
        print("     Real:  ", np.histogram(output[7], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[7], boolean_buckets)[0])
        print("     Acc.: ",  accuracy[7])
        print("Overall accuracy: ", np.sum(accuracy) / 8.0)

    def get_final_stats(self):
        print('this is where we would put final stats.  IF WE HAD ANY')
        #todo present data over time :)
        pass
