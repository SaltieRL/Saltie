from examples.legacy.legacy_input_formatter import LegacyInputFormatter
import numpy as np

from framework.input_formatter.base_input_formatter import BaseInputFormatter


class LSTMInputFormatter(BaseInputFormatter):
    sequence_size = 100
    input_formatter = None

    def __init__(self, input_formatter: BaseInputFormatter, sequence_size=100):
        super().__init__()
        self.sequence_size = sequence_size
        self.input_formatter = input_formatter

    def create_input_array(self, input_array, batch_size=1):
        converted_input_array = self.input_formatter.create_input_array(input_array)
        new_size = batch_size / self.sequence_size
        resultant_array = [int(new_size)] + self.get_input_state_dimension()
        return np.reshape(converted_input_array, resultant_array)

    def get_input_state_dimension(self):
        return [self.sequence_size] + self.input_formatter.get_input_state_dimension()
