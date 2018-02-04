import tensorflow as tf

from conversions.input import tensorflow_input_formatter
from conversions.input.input_formatter import get_state_dim
from modelHelpers.actions import action_factory, dynamic_action_handler
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from trainer.base_classes.base_trainer import BaseTrainer
from trainer.utils.config_objects import *


class DefaultModelTrainer(BaseTrainer):
    OPTIMIZER_CONFIG_HEADER = 'Optimizer Config'
    MISC_CONFIG_HEADER = 'Misc Config'
    action_handler = None
    sess = None
    input_formatter = None
    optimizer = None
    learning_rate = None
    should_apply_features = None
    feature_creator = None
    control_scheme = 'default_scheme'

    def create_config_layout(self):
        super().create_config_layout()
        optimizer_header = self.config_layout.get_header(self.OPTIMIZER_CONFIG_HEADER)
        optimizer_header.add_value('learning_rate', float, "The learning rate for the optimizer")
        optimizer_header.add_value('should_apply_features', bool)  # TODO add description
        self.config_layout.add_header(optimizer_header)

        misc_header = self.config_layout.get_header(self.MISC_CONFIG_HEADER)
        misc_header.add_value('control_scheme', str)  # TODO add description
        self.config_layout.add_header(misc_header)


    def load_config(self):
        super().load_config()
        config = super().create_config()
        try:
            self.learning_rate = config.getfloat(self.OPTIMIZER_CONFIG_HEADER, 'learning_rate')
        except Exception as e:
            self.learning_rate = 0.001
        try:
            self.should_apply_features = config.getboolean(self.OPTIMIZER_CONFIG_HEADER, 'should_apply_features')
        except Exception as e:
            self.should_apply_features = False
        try:
            self.control_scheme = config.get(self.MISC_CONFIG_HEADER, 'control_scheme')
        except Exception as e:
            self.control_scheme = 'default_scheme'

    def setup_trainer(self):
        controls = self.get_field('modelHelpers.actions.action_factory', self.control_scheme)
        self.action_handler = action_factory.get_handler(control_scheme=controls)
        session_config = tf.ConfigProto()
        # session_config.gpu_options.visible_device_list = '1'
        self.sess = tf.Session(config=session_config)
        if self.should_apply_features:
            self.feature_creator = TensorflowFeatureCreator()
        self.input_formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, self.batch_size,
                                                                                   None)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def setup_model(self):
        super().setup_model()
        if self.should_apply_features:
            self.model.apply_feature_creation(self.feature_creator)

    def instantiate_model(self, model_class):
        return model_class(self.sess,
                           self.action_handler.get_logit_size(), action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer, config_file=self.create_config())
