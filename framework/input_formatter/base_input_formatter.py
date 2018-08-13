from rlbot.utils.logging_utils import get_logger


class BaseInputFormatter:
    """
    A skeleton class for all other InputFormatters.
    """

    logger = None

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def create_input_array(self, input_data, batch_size=1):
        """
        Creates an array for the model from the input data.
        :param input_data: Can be any format that the formatter understands.
        :param batch_size: Specifies the batch size of the input data if that is batched.
        Also the batch size what this formatter returns.
        :return: An array that can then be directly fed into a model.
        This can be a python/numpy array or it can be tensorflow/pytorch data.
        """
        return input_data

    def get_input_state_dimension(self):
        """
        :return: An Array representing the input shape
        """
        raise NotImplementedError
