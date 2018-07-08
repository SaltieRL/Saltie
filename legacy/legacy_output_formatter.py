from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LegacyOutputFormatter(BaseOutputFormatter):
    def get_model_output_dimension(self):
        return 8

    def format_model_output(self, output):
        return output[0]
