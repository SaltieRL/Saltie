from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class BaseRewardManager:
    def __init__(self, input_formatter: BaseInputFormatter=None, output_formatter: BaseOutputFormatter=None):
        self.output_formatter = output_formatter
        self.input_formatter = input_formatter

    def has_input_formatter(self):
        return self.input_formatter is not None

    def has_output_formatter(self):
        return self.input_formatter is not None

    def create_reward(self, input_data, output_data, existing_rewards=None, batch_size=1):
        """
        Creates rewards from the given parameters,
        The resulting array needs to be the same length as batch_size
        :param input_data: The data representing the current state that is often given to the model.
        :param output_data: The data representing the output state
        :param existing_rewards:
        :param batch_size:
        :return:
        """
        return existing_rewards
