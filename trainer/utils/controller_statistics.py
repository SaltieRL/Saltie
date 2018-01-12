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

    def __init__(self, packets, model_output, game_tick_packet, input_array, tf_session, action_handler,
                 tutorial_bot=None):
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
        ten = number_prints * .1
        twfive = number_prints * .25
        fifty = number_prints * .5
        names = ["Throttle", "Steer", "Pitch", "Yaw", "Roll", "Jump", "Boost", "Handbrake"]
        print("Every action is printed four times, once all values and then averages over 10%, 25% and 50%")
        for n in range(8):
            print(names[n] + ":")
            print("All:              ", accuracy[n])
            print("Averages every 10%", np.array([average(accuracy[n][int(i * ten):int(i * ten + ten) if not int(i * ten + ten) is int(i * ten) else int(i * ten) + 1]) for i in range(10)]))
            print("               25%", np.array([average(accuracy[n][int(i * twfive):int(i * twfive + twfive) if not int(i * twfive + twfive) is int(i * twfive) else int(i * twfive) + 1]) for i in range(4)]))
            print("               50%", np.array([average(accuracy[n][int(i * fifty):int(i * fifty + fifty) if not int(i * fifty + fifty) is int(i * fifty) else int(i * fifty) + 1]) for i in range(2)]))
