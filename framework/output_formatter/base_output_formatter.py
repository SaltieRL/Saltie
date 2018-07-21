from rlbot.utils.logging_utils import get_logger


class BaseOutputFormatter:

    logger = None

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def format_model_output(self, output):
        return output

    def create_array_for_training(self, output_array, batch_size=1):
        return output_array

    def get_model_output_dimension(self):
        raise NotImplementedError()
