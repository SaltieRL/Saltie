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
    requires_output = False

    def __init__(self, packets, model_output, game_tick_packet, input_array, tf_session, action_handler,
                 bot=None):
        self.sess = tf_session
        self.packets = packets
        self.game_tick_packet = game_tick_packet
        self.input_array = input_array
        self.packet_generator = random_packet_creator.TensorflowPacketGenerator(packets)
        self.tutorial_bot = bot
        self.model_output = model_output
        self.actionHandler = action_handler

        if self.tutorial_bot is None:
            self.requires_output = True

    def create_model(self):
        # clear history
        self.accuracy_over_time = []
        self.bot_data_over_time = []

    def get_amounts(self, bot_output=None):
        controls = tf.transpose(
            self.actionHandler.create_tensorflow_controller_from_selection(self.model_output, self.packets))
        if not self.requires_output:
            bot_output = self.tutorial_bot.get_output_vector(self.game_tick_packet)

        output = self.sess.run(controls)

        accuracy = np.sum(np.isclose(output, bot_output, 0.01), 1) / np.size(output[1])
        self.accuracy_over_time.append(accuracy)
        self.bot_data_over_time.append((output, bot_output))

        analog_buckets = [-1.0001, -0.50001, -0.0001, 0.0001, 0.50001, 1.0001]
        boolean_buckets = [-0.001, 0.50001, 1.0001]
        np.set_printoptions(formatter={'int': '{0:5}'.format})
        names = ["Throttle", "Steer", "Pitch", "Yaw", "Roll", "Jump", "Boost", "Handbrake"]
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, -0>, [0], <0+, 0.5], <0.5, 1]")
        print("Real is model output, Expt is tutorialbot output and Acc. is accuracy")
        for i in range(8):
            print("From here the ranges are [0.0, 0.5>, [0.5, 1.0]") if i is 5 else None
            print(names[i] + ":")
            buckets = analog_buckets if i < 5 else boolean_buckets
            print("     Real:  ", np.histogram(output[i], buckets)[0])
            print("     Expt:  ", np.histogram(bot_output[i], buckets)[0])
            print("     Acc.:  ", accuracy[i])
        print("Overall accuracy: ", np.sum(accuracy) / 8.0)

    def get_final_stats(self):
        def average(numbers):
            return sum(numbers) / len(numbers)

        number_prints = len(self.accuracy_over_time)
        accuracy = np.transpose(self.accuracy_over_time)
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        percentages = [10, 25, 50]
        names = ["Throttle", "Steer", "Pitch", "Yaw", "Roll", "Jump", "Boost", "Handbrake"]
        print("Every action is printed multiple times, once all values and then averages over percentages")
        for n in range(8):
            print(names[n] + ":")
            print("All:              ", accuracy[n])
            for p in percentages:
                r = int(100 / p)
                step = int(number_prints * p / 100)
                print(str(p) + "%:", np.array([average(accuracy[n][int(i * step):int(i * step + step) if not int(i * step + step) is int(i * step) else int(i * step) + 1]) for i in range(r)]))
