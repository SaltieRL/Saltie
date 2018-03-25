import importlib
import inspect
import tensorflow as tf
from bot_code.conversions import output_formatter
from bot_code.models.base_model import BaseAgentModel
from bot_code.trainer.utils.floating_setup import floating_setup
from bot_code.utils.dynamic_import import get_class
import time

class FakeModel(BaseAgentModel):
    """
    An adapter to run teachers (like TutorialBot) while
    implementing the API for BaseAgentModel.
    """
    teacher_package = None
    teacher_class_name = None

    def __init__(self, session, num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, num_actions,
                         input_formatter_info=input_formatter_info,
                         player_index=player_index,
                         action_handler=action_handler,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

    def load_config_file(self):
        super().load_config_file()
        self.teacher_package = self.config_file.get('teacher_package')
        self.teacher_class_name = self.config_file.get('teacher_class_name')
        self.should_float = self.config_file.get('make_player_float')

    def sample_action(self, input_state):
        if self.should_float:
            # This is kinda a hack due to it being unnatural to make the player float.
            # Therefore I'm writing it self-contained, rather than putting properties on this class.
            if not hasattr(self, 'float_location'): self.float_location = [-222, 0, 200 * (1 + self.player_index)]
            if not hasattr(self, 'last_rotation_modification'): self.last_rotation_modification = {}  # player_index -> time of last change of rotation/angular vel
            reset_period = 2.0 # seconds
            now = time.clock()
            if now - self.last_rotation_modification.get(self.player_index, 0) > reset_period:
                self.last_rotation_modification[self.player_index] = now
                floating_setup.set_random_pitch_and_pitch_vel(self.player_index)
                self.float_location[0] += -400 if self.float_location[0] > -300 else 400  # Change position evertime we reset
            floating_setup.make_player_float(self.player_index, self.float_location)

            floating_setup.make_ball_float()
        result = self.sess.run(self.actions, feed_dict={self.input_placeholder: input_state})[0]
        return result

    def get_input(self, model_input=None):
        return self.input_placeholder

    def _create_model(self, model_input):
        teacher_class = get_class(self.teacher_package, self.teacher_class_name)
        teacher = teacher_class(self.batch_size)

        state_object = output_formatter.get_advanced_state(tf.transpose(model_input))

        real_output = teacher.get_output_vector_model(state_object)
        # real_output[0] = tf.Print(real_output[0], real_output, summarize=1)
        self.actions = self.action_handler.create_action_indexes_graph(tf.stack(real_output, axis=1), batch_size=1)
        return None, None

    def create_savers(self):
        # do not create any savers
        pass

    def create_model_hash(self):
        return int(hex(hash(str(self.teacher_package))), 16) % 2 ** 64
