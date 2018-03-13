import tensorflow as tf

from bot_code.conversions.input import tensorflow_input_formatter
from bot_code.modelHelpers.actions import action_factory
from bot_code.modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from bot_code.trainer.base_classes.base_agent_trainer import BaseAgentTrainer
from bot_code.utils.dynamic_import import get_field


class DefaultModelTrainer(BaseAgentTrainer):
    OPTIMIZER_CONFIG_HEADER = 'Optimizer Config'
    MISC_CONFIG_HEADER = 'Misc Config'
    action_handler = None
    sess = None  # The tensorflow session
    input_formatter = None
    optimizer = None
    learning_rate = None
    should_apply_features = None
    feature_creator = None
    control_scheme = 'default_scheme'

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
        controls = get_field('modelHelpers.actions.action_factory', self.control_scheme)
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
                           optimizer=self.optimizer, config_file=self.create_model_config())
