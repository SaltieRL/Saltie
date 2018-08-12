from framework.input_formatter.base_input_formatter import BaseInputFormatter


class HostInputFormatter(BaseInputFormatter):
    """Wraps around an existing input formatter"""

    input_formatter = None

    def __init__(self, input_formatter: BaseInputFormatter):
        super().__init__()
        self.input_formatter = input_formatter

    def create_input_array(self, input_array, batch_size=1):
        converted_input_array = self.input_formatter.create_input_array(input_array)
        return converted_input_array

    def get_input_state_dimension(self):
        return self.input_formatter.get_input_state_dimension()
