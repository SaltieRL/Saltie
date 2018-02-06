import configparser
import importlib
import inspect

import os

from trainer.utils.config_objects import *

from trainer.utils.ding import ding


class BaseTrainer:
    model_class = None
    BASE_CONFIG_HEADER = 'Trainer Configuration'
    MODEL_CONFIG_HEADER = 'Model Configuration'
    batch_size = None
    model = None
    config = None
    config_path = None
    config_layout = None

    def __init__(self, config_path=None, config=None, load_config=True):
        if load_config:
            if config_path is None:
                config_path = os.path.dirname(
                    os.path.dirname(os.path.realpath(__file__))) + os.sep + "configs" + os.sep + self.get_config_name()
                print("Using config at", config_path)
            self.config_path = config_path
            if config is not None:
                print("Using custom config")
                self.config = config
            self.load_config()
        self.create_config_layout()

    def get_class(self, class_package, class_name):
        class_package = importlib.import_module(class_package)
        module_classes = inspect.getmembers(class_package, inspect.isclass)
        for class_group in module_classes:
            if class_group[0] == class_name:
                return class_group[1]
        return None

    def get_field(self, class_package, class_name):
        class_package = importlib.import_module(class_package)
        module_classes = inspect.getmembers(class_package)
        for class_group in module_classes:
            if class_group[0] == class_name:
                return class_group[1]
        return None

    def get_config_name(self):
        return None

    def create_config(self):
        if self.config is None:
            self.config = configparser.RawConfigParser()
            self.config.read(self.config_path)
        return self.config

    def create_config_layout(self):
        self.config_layout = ConfigObject()

        self.config_layout.add_header_name(self.BASE_CONFIG_HEADER)

        model_header = self.config_layout.ConfigHeader(self.MODEL_CONFIG_HEADER)
        model_header.add_value('batch_size', int, default=5000, description="The batch size for training")
        model_header.add_value('model_package', str, description="The package containing the model")
        model_header.add_value('model_name', str, description="The name of the model class")
        self.config_layout.add_header(model_header)
        return self.config_layout

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
        self.model.add_summary_writer(self.get_event_filename())

    def get_event_filename(self):
        return 'event'

    def finish_trainer(self):
        ding()

    def run_trainer(self):
        self._run_trainer()
        self.finish_trainer()

    def _run_trainer(self):
        pass
