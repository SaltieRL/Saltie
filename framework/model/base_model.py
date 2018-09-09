from rlbot.utils.logging_utils import get_logger

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


def native(method):
    method.is_native = True
    return method


class BaseModel:
    """The base model, this will internally hold different tensorflow/keras models"""

    logger = None

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def create_input_layer(self, input_formatter: BaseInputFormatter):
        """Creates the input layer of the model, takes in feeding dicts
        :return The layer you want to feed into hidden layers"""
        pass

    def create_hidden_layers(self, input_layer=None):
        """Creates the internal hidden layers if needed.
        :param input_layer: The previous input layer, can be None
        :return The layer you want to feed into the output layer"""
        pass

    def create_output_layer(self, formatter: BaseOutputFormatter, hidden_layer=None):
        """Creates the output layer of the model.
        :param formatter: Formats the output
        :param hidden_layer: The previous hidden layer, can be None
        :return The output layer of the model"""

    def finalize_model(self):
        """Finalizes the model"""
        pass

    @native
    def fit(self, x, y, rewards=None, batch_size=1):
        pass

    @native
    def predict(self, arr):
        pass

    def save(self, file_path):
        raise NotImplementedError

    def load(self, file_path):
        raise NotImplementedError
