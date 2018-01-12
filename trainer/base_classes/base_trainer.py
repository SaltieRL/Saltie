import configparser
import importlib
import inspect

import os
import tensorflow as tf

from trainer.utils.ding import ding


class BaseTrainer:
    model_class = None
    BASE_CONFIG_HEADER = 'Trainer Configuration'
    MODEL_CONFIG_HEADER = 'Model Configuration'
    batch_size = None
    model = None
    config = None

    def __init__(self):
        self.load_config()

    def get_class(self, class_package, class_name):
        class_package = importlib.import_module(class_package)
        module_classes = inspect.getmembers(class_package, inspect.isclass)
        for class_group in module_classes:
            if class_group[0] == class_name:
                return class_group[1]
        return None

    def get_config_name(self):
        return None

    def create_config(self):
        if self.config is None:
            self.config = configparser.RawConfigParser()
            file = 'configs/' + str(self.get_config_name())
            dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            self.config.read(dir_path + '/' + file)
        return self.config

    def load_config(self):
        # Obtaining necessary data for training from the config
        config = self.create_config()
        self.batch_size = config.getint(self.MODEL_CONFIG_HEADER, 'batch_size')

        # Over here the model data is obtained
        model_package = config.get(self.MODEL_CONFIG_HEADER, 'model_package')
        model_name = config.get(self.MODEL_CONFIG_HEADER, 'model_name')
        self.model_class = self.get_class(model_package, model_name)

    def setup_trainer(self):
        """Called to setup the functions of the trainer and anything needed for the creation of the model"""
        pass

    def instantiate_model(self, model_class):
        return model_class()

    def setup_model(self):
        self.model = self.instantiate_model(self.model_class)
        self.model.summary_writer = tf.summary.FileWriter(
            self.model.get_event_path(self.get_event_filename()))

    def get_event_filename(self):
        return 'event'

    def finish_trainer(self):
        ding()

    def run_trainer(self):
        self._run_trainer()
        self.finish_trainer()

    def _run_trainer(self):
        pass
