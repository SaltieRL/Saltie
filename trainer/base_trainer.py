from framework.model_holder.base_model_holder import BaseModelHolder


class BaseTrainer:

    def __init__(self, model_holder: BaseModelHolder):
        self.model_holder = model_holder
        model_holder.initialize_model(load=True)
