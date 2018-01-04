from TutorialBot import RandomTFArray, tensorflow_input_formatter, tutorial_bot_output
import numpy as np


class OutputChecks:
    def __init__(self, packets, model, tfsession, actionHandler):
        self.sess = tfsession
        self.packets = packets
        self.formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, packets)
        self.packet_generator = RandomTFArray.TensorflowPacketGenerator(packets)
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

        controls = self.actionHandler.create_tensorflow_controller_output_from_actions(self.model.argmax, self.packets)

        output, bot_output = self.sess.run([controls, self.tutorial_bot.get_output_vector(game_tick_packet)])
        transposed = np.matrix.transpose(output)  # transposed[0] gives all the returned values for throttle sorted

        buckets = [-1.0, -0.5, -0.0001, 0.0001, 0.5111111, 1]
        print("Splitting up everything in ranges: [-1, -0.5>, [-0.5, 0->, [0], <0+, 0.5], <0.5, 1] and the second array is for the bot output")
        print("Throttle :  ")
        print("     Real:  ", np.histogram(transposed[0], buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[0], buckets)[0])
        print("Steer    :  ")
        print("     Real:  ", np.histogram(transposed[1], buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[1], buckets)[0])
        print("Pitch    :  ")
        print("     Real:  ", np.histogram(transposed[2], buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[2], buckets)[0])
        print("Yaw      :  ")
        print("     Real:  ", np.histogram(transposed[3], buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[3], buckets)[0])
        print("Roll     :  ")
        print("     Real:  ", np.histogram(transposed[4], buckets)[0])
        print("     Expt:  ", np.histogram(bot_output[4], buckets)[0])
        print("From here the ranges are [0.0, 0.5>, [0.5, 1.0]")
        print("Jump     :  ")
        print("     Real:  ", np.histogram(transposed[5], [0, 0.5, 1])[0])
        print("     Expt:  ", np.histogram(bot_output[5], [0, 0.5, 1])[0])
        print("Boost    :  ")
        print("     Real:  ", np.histogram(transposed[6], [0, 0.5, 1])[0])
        print("     Expt:  ", np.histogram(bot_output[6], [0, 0.5, 1])[0])
        print("Handbrake:  ")
        print("     Real:  ", np.histogram(transposed[7], [0, 0.5, 1])[0])
        print("     Expt:  ", np.histogram(bot_output[7], [0, 0.5, 1])[0])


