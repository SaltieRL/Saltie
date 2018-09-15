import numpy as np
import pandas as pd
from carball.analysis.utils.pandas_manager import PandasManager

from framework.data_generator.base_generator import BaseDataGenerator
from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class LocalCacheCreator:

    def __init__(self, input_formatter: BaseInputFormatter, output_formatter: BaseOutputFormatter, data_generator: BaseDataGenerator):
        self.data_generator = data_generator
        self.output_formatter = output_formatter
        self.input_formatter = input_formatter
        self.cache = None

    def create_cache(self):
        input_array = np.array([])
        output_array = np.array([])

        for data in self.data_generator.get_data():
            np.append(input_array, np.array(self.input_formatter.create_input_array(data)))
            np.append(output_array, np.array(self.output_formatter.create_array_for_training(data)))

        self.cache = pd.DataFrame(data={"input": input_array, "output": output_array})

    def save_cache(self, file_path):
        result = PandasManager.safe_write_pandas_to_memory(self.cache)
        with open(file_path, 'w') as f:
            f.write(result)
