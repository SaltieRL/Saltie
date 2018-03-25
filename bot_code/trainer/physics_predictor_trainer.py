import tensorflow as tf

from bot_code.conversions.input import tensorflow_input_formatter
from bot_code.modelHelpers.actions import action_factory
from bot_code.trainer.base_classes.base_trainer import BaseTrainer
from bot_code.trainer.utils.ding import text_to_speech


class PhysicsInputOutputGenerator():
    def get_input_dim(self):
        raise NotImplementedError('Derived classes must override this.')
    def get_output_dim(self):
        raise NotImplementedError('Derived classes must override this.')

class DummyPhysicsInputOutputGenerator(PhysicsInputOutputGenerator):
    def get_input_dim(self):
        return 2
    def get_output_dim(self):
        return 1

class PhysicsPredictorTrainer(BaseTrainer):
    '''
    A class that trains models that predict physics.
    Essentially, building a regression function over f(state, player_input, dt) -> state
    '''

    OPTIMIZER_CONFIG_HEADER = 'Optimizer Config'
    sess = None  # The tensorflow session
    # input_formatter = None
    optimizer = None
    learning_rate = None

    inputOutputGenerator = DummyPhysicsInputOutputGenerator()

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
        session_config = tf.ConfigProto()
        # session_config.gpu_options.visible_device_list = '1'
        self.sess = tf.Session(config=session_config)
        # self.input_formatter = tensorflow_input_formatter.TensorflowInputFormatter(0, 0, self.batch_size, None)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def instantiate_model(self, model_class):
        # model_class is assumed to be a subclass of
        # bot_code.models.physics_predictor.physics_predictor.PhysicsPredictor
        return model_class(
            self.sess,
            input_dim=self.inputOutputGenerator.get_input_dim(),
            output_dim=self.inputOutputGenerator.get_output_dim(),
        )

    def setup_model(self):
        super().setup_model()

    def _run_trainer(self):
        print('Totally training over here')

    def finish_trainer(self):
        print ("done.")
        text_to_speech('Training finished.')




if __name__ == '__main__':
    PhysicsPredictorTrainer().run()
