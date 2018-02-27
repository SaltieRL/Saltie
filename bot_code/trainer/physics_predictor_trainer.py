import tensorflow as tf

from bot_code.conversions.input import tensorflow_input_formatter
from bot_code.modelHelpers.actions import action_factory
from bot_code.modelHelpers.tensorflow_feature_creator import TensorflowFeatureCreator
from bot_code.trainer.base_classes.base_trainer import BaseTrainer

class PhysicsPredictorTrainer(BaseTrainer):
    '''
    A class that trains models that predict physics.
    Essentially, building a regression function over f(state, player_input, dt) -> state
    '''

    OPTIMIZER_CONFIG_HEADER = 'Optimizer Config'
    MISC_CONFIG_HEADER = 'Misc Config'
    sess = None  # The tensorflow session
    input_formatter = None
    optimizer = None
    learning_rate = None
    should_apply_features = None
    feature_creator = None


    def get_config_name(self):
        return 'physics_predictor_trainer.cfg'
    def load_config(self):
        super().load_config()
        config = super().create_config()
        try:
            self.learning_rate = config.getfloat(self.OPTIMIZER_CONFIG_HEADER, 'learning_rate')
        except Exception as e:
            self.learning_rate = 0.001

    def setup_trainer(self):
        controls = self.get_field('modelHelpers.actions.action_factory', self.control_scheme)
        self.action_handler = action_factory.get_handler(control_scheme=controls)
        session_config = tf.ConfigProto()
        # session_config.gpu_options.visible_device_list = '1'
        self.sess = tf.Session(config=session_config)
        if self.should_apply_features:
            self.feature_creator = TensorflowFeatureCreator()
        self.input_formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, self.batch_size, None)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def setup_model(self):
        super().setup_model()
        if self.should_apply_features:
            self.model.apply_feature_creation(self.feature_creator)

    def _run_trainer(self):
        print('Totally training over here')

if __name__ == '__main__':
    PhysicsPredictorTrainer().run()
