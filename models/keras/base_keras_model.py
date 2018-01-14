from models.base_model import BaseModel


class BaseKerasModel(BaseModel):
    def __init__(self, session, state_dim, num_actions, player_index=-1, action_handler=None, is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), summary_writer=None, summary_every=100,
                 config_file=None):
        super().__init__(session, state_dim, num_actions, player_index, action_handler, is_training, optimizer,
                         summary_writer, summary_every, config_file)

    def printParameters(self):
        super().printParameters()

    def _create_variables(self):
        pass

    def sample_action(self, input_state):
        return super().sample_action(input_state)

    def create_copy_training_model(self, model_input=None, taken_actions=None):
        return super().create_copy_training_model(model_input, taken_actions)

    def get_input(self, model_input=None):
        # given maybe input return keras version  for your model
        # note that the super class uses a tensorflow placeholder
        return super().get_input(model_input)

    def _create_model(self, model_input):
        return super()._create_model(model_input)

    def _initialize_variables(self):
        super()._initialize_variables()

    def run_train_step(self, calculate_summaries, input_states, actions):
        super().run_train_step(calculate_summaries, input_states, actions)

    def _add_summary_writer(self):
        super()._add_summary_writer()

    def load_config_file(self):
        super().load_config_file()

    def add_saver(self, name, variable_list):
        super().add_saver(name, variable_list)

    def create_savers(self):
        super().create_savers()

    def _save_model(self, session, saver, file_path, global_step):
        super()._save_model(session, saver, file_path, global_step)

    def _load_model(self, session, saver, path):
        super()._load_model(session, saver, path)

    def create_model_hash(self):
        return super().create_model_hash()
