import importlib
import inspect
import tensorflow as tf
from conversions import output_formatter
from models.base_model import BaseModel


class FakeModel(BaseModel):
    teacher_package = None

    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def get_class(self, class_package, class_name):
        class_package = importlib.import_module(class_package)
        module_classes = inspect.getmembers(class_package, inspect.isclass)
        for class_group in module_classes:
            if class_group[0] == class_name:
                return class_group[1]
        return None

    def load_config_file(self):
        super().load_config_file()
        self.teacher_package = self.config_file.get('Model Configuration', 'teacher_package')

    def sample_action(self, input_state):
        result = self.sess.run(self.actions, feed_dict={self.input_placeholder: input_state})[0]
        # print(result)
        result = [int(x) for x in result]
        return result

    def get_input(self, model_input=None):
        return self.input_placeholder

    def _create_model(self, model_input):
        output_creator = self.get_class(self.teacher_package, 'TutorialBotOutput')(self.batch_size)

        game_tick_packet = output_formatter.get_advanced_state(tf.transpose(model_input))

        real_output = output_creator.get_output_vector(game_tick_packet)
        # real_output[0] = tf.Print(real_output[0], real_output, summarize=1)
        self.actions = self.action_handler.create_action_indexes_graph(tf.stack(real_output, axis=1), batch_size=1)
        return None, None

    def create_savers(self):
        # do not create any savers
        pass

    def create_model_hash(self):
        return int(hex(hash(str(self.teacher_package))), 16) % 2 ** 64
