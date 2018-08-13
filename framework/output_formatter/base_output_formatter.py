from rlbot.utils.logging_utils import get_logger


class BaseOutputFormatter:

    logger = None

    def __init__(self):
        self.logger = get_logger(str(type(self).__name__))

    def format_model_output(self, output, batch_size=1):
        """
        Takes in the raw data from the model and converts it into a format for prediction.
        :param output: Raw data from the model, typically an array
        :param batch_size: Specifies the batch size of the output data if that is batched.
        Also the batch size what this should formatter returns.
        :return: The format needed.  This can be anything.
        """
        return output

    def create_array_for_training(self, predicted_data, batch_size=1):
        """
        Converts data that is what the model is training against into a format the model can understand.
        :param predicted_data: This can be any format.
        :param batch_size: Specifies the batch size of the predicted data.
        :return:  This should be a format that tensorflow/pytorch can directly ingest.
        """
        return predicted_data

    def get_model_output_dimension(self):
        """
        This is the dimension expected by 'format_model_output'
        :return: Result need to be an array specifying the output dimensions.
        ex: [8] or [5, 6]
        """
        raise NotImplementedError()
