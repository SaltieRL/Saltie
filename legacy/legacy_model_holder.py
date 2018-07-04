from framework.model_holder.base_model_holder import BaseModelHolder


class LegacyModelHolder(BaseModelHolder):

    def process_pair(self, input_array, controller_array, pair_number, hashed_name):
        self.train_step(input_array, controller_array)
