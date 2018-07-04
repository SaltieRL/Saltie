from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


def native(method):
    method.is_native = True
    return method


class BaseModel:
    """The base model, this will internally hold different tensorflow/keras models"""

    def create_input_layer(self, input_formatter: BaseInputFormatter):
        """Creates the input layer of the model, takes in feeding dicts"""
        pass

    def create_hidden_layers(self):
        """Creates the internal hidden layers if needed"""
        pass

    def create_output_layer(self, formatter: BaseOutputFormatter):
        """Creates the output layer of the model.
        :param formatter:
        :return The output layer of the model"""

    @native
    def fit(self, x, y):
        pass

    def predict(self, arr):
        pass
