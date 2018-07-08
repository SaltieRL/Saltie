from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter
from framework.model.base_model import BaseModel


class BaseModelHolder:

    use_custom_fit = False
    use_custom_sample_action = False
    model_output = None

    def __init__(self, model: BaseModel, input_formatter: BaseInputFormatter, output_formatter: BaseOutputFormatter):
        self.model = model
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter

        self.use_custom_fit = not hasattr(self.model.fit, 'is_native')
        self.use_custom_sample_action = not hasattr(self.model.predict, 'is_native')

    def initialize_model(self):
        self.model.create_input_layer(self.input_formatter)
        self.model.create_hidden_layers()
        self.model_output = self.model.create_output_layer(self.output_formatter)
        self.model.finalize_model()

    def train_step(self, input_array, output_array):
        arr = self.input_formatter.create_input_array(input_array)
        out = self.output_formatter.create_array_for_training(output_array)
        if self.use_custom_fit:
            self.model.fit(arr, out)

    def predict(self, prediction_input):
        """
        Predicts an output given the input
        :param prediction_input: The input, this can be anything as it will go through a BaseInputFormatter 
        :return:
        """
        arr = self.input_formatter.create_input_array(prediction_input)
        if self.use_custom_sample_action:
            output = self.model.predict(arr)
        else:
            output = self.__predict(arr)
        return self.output_formatter.format_model_output(output)

    def finish_training(self, save_model=True):
        if save_model:
            file_path = self.get_file_path()
            print('saving model at:', file_path)
            self.model.save(file_path)

    def __fit(self, arr, out):
        raise NotImplementedError

    def __predict(self, arr):
        raise NotImplementedError

    def get_file_path(self):
        return 'weights/' + str(type(self.model).__name__) + '.mdl'
