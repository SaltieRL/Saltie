from rlbot.utils.logging_utils import get_logger


class BaseInputFormatter:
    """
    A skeleton class for all other InputFormatters.
    """

    logger = None  # A logger that is named after the class.

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def create_input_array(self, input_data, batch_size=1):
        """
        Creates an array for the model from the input data.
        :param input_data: Can be any format that the formatter understands.
        :param batch_size: Specifies the batch size of the input data if that is batched.
        Also the batch size what this formatter returns.
        :return: An array that can then be directly fed into a model.
        This can be a python/numpy array.
        """
        return input_data

    def transform_tensor(self, input_tensor):
        """
        Takes in a tensor and performs generic transformations or feature creation on it.
        :param input_tensor: A tensorflow/pytorch tensor.
        :return: A tensorflow/pytorch tensor
        """
        raise NotImplementedError

    def get_input_state_dimension(self):
        """
        This is the dimension returned by 'create_input_array'
        :return: Result need to be an array specifying the output dimensions.
        ex: [8] or [5, 6]
        """
        raise NotImplementedError
