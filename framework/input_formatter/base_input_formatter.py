from rlbot.utils.logging_utils import get_logger


class BaseInputFormatter:

    logger = None

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def create_input_array(self, input_array, batch_size=1):
        """
        Creates an array for the model from the game_tick_packet
        :return: A massive array representing that packet
        """
        return input_array

    def get_input_state_dimension(self):
        """
        :return: An Array representing the input shape
        """
        raise NotImplementedError
