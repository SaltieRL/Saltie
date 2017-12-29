from TutorialBot import RandomTFArray, tensorflow_input_formatter, tutorial_bot_output
from conversions import input_formatter
import tensorflow as tf
import numpy as np


class OutputChecks:
    def __init__(self, packets, variablepath, tfsession):
        self.sess = tfsession
        self.packets = packets
        self.formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, packets)
        self.packet_generator = RandomTFArray.TensorflowPacketGenerator(packets)
        self.tutorial_bot = tutorial_bot_output.TutorialBotOutput(packets)
        n_neurons_hidden = 128  # every layer of neurons
        n_input = input_formatter.get_state_dim_with_features()  # data input
        n_output = 8  # total outputs
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_neurons_hidden])),
            'h2': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
            'h3': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
            'h4': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
            'h5': tf.Variable(tf.random_normal([n_neurons_hidden, n_neurons_hidden])),
            'out': tf.Variable(tf.random_normal([n_neurons_hidden, n_output]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_neurons_hidden])),
            'b2': tf.Variable(tf.random_normal([n_neurons_hidden])),
            'b3': tf.Variable(tf.random_normal([n_neurons_hidden])),
            'b4': tf.Variable(tf.random_normal([n_neurons_hidden])),
            'b5': tf.Variable(tf.random_normal([n_neurons_hidden])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, variablepath)
        self.weights = weights
        self.biases = biases

    def get_random_data(self):
        game_tick_packet = self.packet_generator.get_random_array()
        output_array = self.formatter.create_input_array(game_tick_packet)[0]
        # reverse the shape of the array
        return output_array, game_tick_packet

    def get_amounts(self):
        def get_output(x):
            layer_1 = tf.nn.relu6(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
            layer_2 = tf.nn.relu6(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
            layer_3 = tf.nn.relu6(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
            layer_4 = tf.nn.relu6(tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4']))
            layer_5 = tf.nn.relu6(tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5']))
            out_layer = tf.nn.sigmoid(tf.matmul(layer_5, self.weights['out']) + self.biases['out'])
            return out_layer

        input_state, game_tick_packet = self.get_random_data()
        output = self.sess.run(get_output(input_state))
        bot_output = self.sess.run(self.tutorial_bot.get_output_vector(game_tick_packet, output)[1])
        transposed = np.matrix.transpose(output)  # transposed[0] gives all the returned values for throttle sorted
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, 0>, [0, 0.5>, [0.5, 1] and the second array is for the bot output")
        print("Throttle:  ", np.histogram(transposed[0], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[0], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("Steer:     ", np.histogram(transposed[1], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[1], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("Pitch:     ", np.histogram(transposed[2], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[2], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("Yaw:       ", np.histogram(transposed[3], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[3], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("Roll:      ", np.histogram(transposed[4], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[4], [-1.0, -0.5, 0, 0.5, 1])[0])
        print("From here the ranges are [0.0, 0.5>, [0.5, 1.0]")
        print("Jump:      ", np.histogram(transposed[5], [0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[5], [0, 0.5, 1])[0])
        print("Boost:     ", np.histogram(transposed[6], [0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[6], [0, 0.5, 1])[0])
        print("Handbrake: ", np.histogram(transposed[7], [0, 0.5, 1])[0])
        print("           ", np.histogram(bot_output[7], [0, 0.5, 1])[0])


