import tensorflow as tf

from conversions.input import tensorflow_input_formatter
from modelHelpers.actions import action_factory, dynamic_action_handler
from modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from trainer.base_classes.base_trainer import BaseTrainer


class DefaultModelTrainer(BaseTrainer):
    action_handler = None
    sess = None
    formatter = None
    optimizer = None
    learning_rate = 0.0001

    def setup_trainer(self):
        self.action_handler = action_factory.get_handler(control_scheme=dynamic_action_handler.super_split_scheme)
        self.sess = tf.Session()
        feature_creator = TensorflowFeatureCreator()
        self.formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, self.batch_size, feature_creator)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def instantiate_model(self, model_class):
        return model_class(self.sess, self.formatter.get_state_dim_with_features(),
                           self.action_handler.get_logit_size(), action_handler=self.action_handler, is_training=True,
                           optimizer=self.optimizer, config_file=self.create_config())
