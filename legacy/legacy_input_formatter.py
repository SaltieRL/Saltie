from framework.input_formatter.base_input_formatter import BaseInputFormatter


class LegacyInputFormatter(BaseInputFormatter):
    def get_input_state_dimension(self):
        return 219
