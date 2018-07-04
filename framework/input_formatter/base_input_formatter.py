class BaseInputFormatter:
    def create_input_array(self, input_array):
        """
        Creates an array for the model from the game_tick_packet
        :return: A massive array representing that packet
        """
        return input_array

    def get_input_state_dimension(self):
        raise NotImplementedError
