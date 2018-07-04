class BaseInputFormatter:
    def create_input_array(self, pandas, passed_time=0.0):
        """
        Creates an array for the model from the game_tick_packet

        :param pandas: A game packet for a single point in time
        :param passed_time: Time between the last frame and this one
        :return: A massive array representing that packet
        """
        pass

    def create_prediction_array(self, output_array):
        pass

    def create_input_placeholder(self):
        pass

    def get_input_state_dimension(self):
        pass
