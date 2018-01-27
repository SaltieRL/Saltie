from trainer.utils import random_packet_creator
import tensorflow as tf
import numpy as np


class OutputChecks:
    model_output = None
    game_tick_packet = None
    accuracy_over_time = None
    bot_data_over_time = None
    requires_input = False
    requires_bot_output = False
    controls = None

    def __init__(self, tf_session, action_handler, batch_size, model_output,
                 game_tick_packet=None,
                 bot=None,
                 model_placeholder=None):
        self.sess = tf_session
        self.batch_size = batch_size
        self.game_tick_packet = game_tick_packet
        self.tutorial_bot = bot
        self.model_output = model_output
        self.model_input = model_placeholder
        self.actionHandler = action_handler

        if self.tutorial_bot is None:
            self.requires_bot_output = True

        if self.model_input is not None:
            self.requires_input = True

    def create_model(self):
        # clear history
        self.accuracy_over_time = []
        self.bot_data_over_time = []
        self.controls = tf.transpose(
            self.actionHandler.create_tensorflow_controller_from_selection(self.model_output,
                                                                           self.batch_size))

    def get_amounts(self, input_array=None, bot_output=None):

        if not self.requires_bot_output:
            bot_output = self.sess.run(self.tutorial_bot.get_output_vector(self.game_tick_packet))
        else:
            if bot_output is None:
                print("Missing correct output")
                return

        if not self.requires_input:
            output = self.sess.run(self.controls)
        else:
            output = self.sess.run(self.controls, feed_dict={self.model_input: input_array})

        accuracy = np.sum(np.isclose(output, bot_output, 0.2), 1) / np.size(output[1])
        self.accuracy_over_time.append(accuracy)
        self.bot_data_over_time.append((output, bot_output))

        analog_buckets = [-1.0001, -0.50001, -0.1000, 0.1000, 0.50001, 1.0001]
        boolean_buckets = [-0.001, 0.50001, 1.0001]
        np.set_printoptions(formatter={'int': '{0:5}'.format})
        names = ["Throttle", "Steer", "Pitch", "Yaw", "Roll", "Jump", "Boost", "Handbrake"]
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, -0.1>, [0], <0.1+, 0.5], <0.5, 1]")
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
