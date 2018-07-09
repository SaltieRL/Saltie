from legacy.legacy_input_formatter import LegacyInputFormatter
import numpy as np


class TrainingLSTMInputFormatter(LegacyInputFormatter):
    sequence_size = 100

    def create_input_array(self, input_array, batch_size=1):
        new_size = batch_size / self.sequence_size
        return np.reshape(input_array, [int(new_size), self.sequence_size, self.feature_size])

    def get_input_state_dimension(self):
        return [self.sequence_size, self.feature_size]
