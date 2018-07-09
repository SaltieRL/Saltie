from rlbot.utils.logging_utils import get_logger

from framework.input_formatter.base_input_formatter import BaseInputFormatter


class LegacyInputFormatter(BaseInputFormatter):
    feature_size = 219
    logger = None

    def get_input_state_dimension(self):
        return [self.feature_size]
