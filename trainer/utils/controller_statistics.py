from trainer.utils import random_packet_creator
from TutorialBot import tutorial_bot_output
from conversions.input import tensorflow_input_formatter
import tensorflow as tf
import numpy as np


class OutputChecks:
    def __init__(self, packets, model, tfsession, actionHandler):
        self.sess = tfsession
        self.packets = packets
        self.formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, packets)
        self.packet_generator = random_packet_creator.TensorflowPacketGenerator(packets)
        self.tutorial_bot = tutorial_bot_output.TutorialBotOutput(packets)
        self.model = model
        self.actionHandler = actionHandler

    def get_random_data(self):
        game_tick_packet = self.packet_generator.get_random_array()
        output_array = self.formatter.create_input_array(game_tick_packet)[0]
        # reverse the shape of the array
        return output_array, game_tick_packet

    def get_amounts(self):
        input_state, game_tick_packet = self.get_random_data()
        self.model.batch_size = self.packets
        self.model.mini_batch_size = self.packets
        self.model.create_model(input_state)

        controls = tf.transpose(
            self.actionHandler.create_tensorflow_controller_output_from_actions(self.model.argmax, self.packets))
        expected = self.tutorial_bot.get_output_vector(game_tick_packet)

        output, bot_output = self.sess.run([controls, expected])

        accuracy = np.sum(np.isclose(output, bot_output, 0.01), 1) / np.size(output[1])

        analog_buckets = [-1.0001, -0.50001, -0.0001, 0.0001, 0.50001, 1.0001]
        boolean_buckets = [-0.001, 0.50001, 1.0001]
        np.set_printoptions(formatter={'int': '{0:5}'.format})
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, 0->, [0], <0+, 0.5], <0.5, 1]")
        print("Real is model output, Expt is tutorialbot output and Acc. is accuracy")
        print("Throttle :  ")
        print("     Real:  ", np.histogram(output[0], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[0], analog_buckets)[0])
        print("     Acc.: ", accuracy[0])
        print("Steer    :  ")
        print("     Real:  ", np.histogram(output[1], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[1], analog_buckets)[0])
        print("     Acc.: ", accuracy[1])
        print("Pitch    :  ")
        print("     Real:  ", np.histogram(output[2], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[2], analog_buckets)[0])
        print("     Acc.: ", accuracy[2])
        print("Yaw      :  ")
        print("     Real:  ", np.histogram(output[3], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[3], analog_buckets)[0])
        print("     Acc.: ", accuracy[3])
        print("Roll     :  ")
        print("     Real:  ", np.histogram(output[4], analog_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[4], analog_buckets)[0])
        print("     Acc.: ", accuracy[4])
        print("From here the ranges are [0.0, 0.5>, [0.5, 1.0]")
        print("Jump     :  ")
        print("     Real:  ", np.histogram(output[5], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[5], boolean_buckets)[0])
        print("     Acc.:  ", accuracy[5])
        print("Boost    :  ")
        print("     Real:  ", np.histogram(output[6], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[6], boolean_buckets)[0])
        print("     Acc.: ", accuracy[6])
        print("Handbrake:  ")
        print("     Real:  ", np.histogram(output[7], boolean_buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[7], boolean_buckets)[0])
        print("     Acc.:  ", accuracy[7])
        print("Overall accuracy: ", np.sum(accuracy) / 8)
