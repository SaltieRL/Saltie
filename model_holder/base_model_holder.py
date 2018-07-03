from input_formatter.base_input_formatter import BaseInputFormatter
from model.base_model import BaseModel


class BaseModelHolder:

    def __init__(self, model: BaseModel, input_formatter: BaseInputFormatter):
        self.model = model
        self.input_formatter = input_formatter

    def process_pair(self, input_array, output_array, pair_number, hashed_name):
        print(input_array, output_array, pair_number, hashed_name)
