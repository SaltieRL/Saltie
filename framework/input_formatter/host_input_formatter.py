from framework.input_formatter.base_input_formatter import BaseInputFormatter


class HostInputFormatter(BaseInputFormatter):
    """Wraps around an existing input formatter"""

    input_formatter = None

    def __init__(self, input_formatter: BaseInputFormatter):
        super().__init__()
        self.input_formatter = input_formatter

    def create_input_array(self, input_array, batch_size=1):
        """
        By default just calls the input_formatter this holds.
        Can be used to chain together formatting.
        :param input_array: Can be any format that the formatter understands.
        :param batch_size: Specifies the batch size of the input data if that is batched.
        Also the batch size what this formatter returns.
        :return: An array that can then be directly fed into a model.
        This can be a python/numpy array or it can be tensorflow/pytorch data.
        """
        converted_input_array = self.input_formatter.create_input_array(input_array)
        return converted_input_array

    def get_input_state_dimension(self):
        return self.input_formatter.get_input_state_dimension()
