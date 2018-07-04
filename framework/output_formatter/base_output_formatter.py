class BaseOutputFormatter:

    def format_model_output(self, output):
        return output

    def create_array_for_training(self, output_array):
        return output_array

    def get_model_output_dimension(self):
        raise NotImplementedError
