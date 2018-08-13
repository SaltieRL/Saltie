import numpy as np
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LSTMOutputFormatter(BaseOutputFormatter):

    sequence_size = 100

    def __init__(self, output_formatter: BaseOutputFormatter, sequence_size=100):
        super().__init__()
        self.sequence_size = sequence_size
        self.output_formatter = output_formatter

    def create_array_for_training(self, predicted_data, batch_size=1):
        new_size = batch_size / self.sequence_size
        result_array = [int(new_size), self.sequence_size] + self.get_model_output_dimension()
        return np.reshape(predicted_data, result_array)

    def get_model_output_dimension(self):
        return self.output_formatter.get_model_output_dimension()

    def format_model_output(self, output):
        return output[0][0]
