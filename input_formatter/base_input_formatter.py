class BaseInputFormatter:
    def create_input_array(self, game_tick_packet, passed_time=0.0):
        """
        Creates an array for the model from the game_tick_packet

        :param game_tick_packet: A game packet for a single point in time
        :param passed_time: Time between the last frame and this one
        :return: A massive array representing that packet
        """
        pass
