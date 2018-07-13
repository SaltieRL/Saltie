from framework.model_holder.base_model_holder import BaseModelHolder


class AutoencoderModelHolder(BaseModelHolder):

    def train_step(self, input_array, output_array, batch_size=1):
        arr = self.input_formatter.create_input_array(input_array, batch_size)
        self.model.fit(arr, arr)

    def process_pair(self, input_array, controller_array, pair_number, hashed_name, batch_size=1):
        self.train_step(input_array, controller_array, batch_size)
