from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class HostOutputFormatter(BaseOutputFormatter):
    """Wraps around an output formatter"""

    def __init__(self, output_formatter: BaseOutputFormatter):
        super().__init__()
        self.output_formatter = output_formatter

    def create_array_for_training(self, output_array, batch_size=1):
        return self.output_formatter.create_array_for_training(output_array, batch_size)

    def get_model_output_dimension(self):
        return self.output_formatter.get_model_output_dimension()

    def format_model_output(self, output):
        return self.output_formatter.format_model_output(output)
