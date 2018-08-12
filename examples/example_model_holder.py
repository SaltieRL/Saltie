from framework.model_holder.base_model_holder import BaseModelHolder


class ExampleModelHolder(BaseModelHolder):

    def process_pair(self, input_array, controller_array, pair_number, hashed_name, batch_size=1):
        output = self.predict(input_array)
        print(output)
        self.train_step(input_array, controller_array, batch_size=batch_size)
