from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class AutoencoderOutputFormatter(BaseOutputFormatter):
    def __init__(self, input_formatter: BaseInputFormatter):
        super().__init__()
        self.input_formatter = input_formatter

    def get_model_output_dimension(self):
        return self.input_formatter.get_input_state_dimension()
