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
        Creates one or multiple numpy arrays for the model from the input data.
        :param input_data: Can be any format that the formatter understands.
        :param batch_size: Specifies the batch size of the input data if that is batched.
        Also the batch size what this formatter returns.
        :return: One or multiple numpy arrays that can then be directly fed into a model.
        The numpy arrays have the shapes returned by 'get_input_state_dimension'.
        The numpy arrays are grouped in a python array in the same order as the shapes.
        ex: [np.zeros((1, 8,))]
        ex: [np.zeros((1, 5, 6))]
        ex: [np.zeros((1, 3, 9)), np.zeros((1, 5,))]
        """
        return input_data

    def transform_tensor(self, input_tensor):
        """
        Takes in a tensor and performs generic transformations or feature creation on it.
        :param input_tensor: A tensorflow/pytorch tensor.
        :return: A tensorflow/pytorch tensor
        """
        return input_tensor

    def get_input_state_dimension(self):
        """
        This returns the shapes of the numpy arrays returned by 'create_input_array'.
        :return: Result needs to be an array of shapes.
        Shapes are tuples that contain the dimensions of an array.
        ex: [(8,)] or [(5, 6)] or [(3, 9), (5,)]
        """
        raise NotImplementedError
