from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.input_formatter.output_formatter.base_output_formatter import BaseOutputFormatter
from framework.model.base_model import BaseModel


class BaseModelHolder:

    use_custom_fit = False
    use_custom_sample_action = False
    model_output = None

    def __init__(self, model: BaseModel, input_formatter: BaseInputFormatter, output_formatter: BaseOutputFormatter):
        self.model = model
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter

    def initialize_model(self):
        input_placeholder = self.input_formatter.create_input_placeholder()
        self.model.create_input_layer(input_placeholder)
        self.model.create_hidden_layers()
        self.model_output = self.model.create_output_layer()
        self.use_custom_fit = not hasattr(self.model.fit, 'is_native')
        self.use_custom_sample_action = not hasattr(self.model.predict, 'is_native')

    def train_step(self, input_array, output_array):
        arr = self.input_formatter.create_input_array(input_array)
        out = self.input_formatter.create_output_array(output_array)
        if self.use_custom_fit:
            self.model.fit(arr, out)

    def predict(self, input_array):
        arr = self.input_formatter.create_input_array(input_array)
        if self.use_custom_sample_action:
            output = self.model.predict(arr)
        else:
            output = self.__predict(arr)
        return self.output_formatter.format_output(output)

    def __fit(self, arr, out):
        raise NotImplementedError

    def __predict(self, arr):
        raise NotImplementedError

