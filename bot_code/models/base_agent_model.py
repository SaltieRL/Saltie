import tensorflow as tf

from bot_code.modelHelpers import tensorflow_feature_creator
from bot_code.models.base_model import BaseModel
from bot_code.conversions.input.input_formatter import InputFormatter


class BaseAgentModel(BaseModel):
    """
    A base class for any model that outputs car actions given the the current state (+features).
    """
    def __init__(self,
                 session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 is_training=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 summary_writer=None,
                 summary_every=100,
                 config_file=None):
        # TODO: Should index here always == player_index?
        (team, index) = input_formatter_info[0], input_formatter_info[1]

        self.input_formatter = InputFormatter(team, index)
        self.state_dim = self.input_formatter.get_state_dim()
        self.state_feature_dim = self.state_dim
        self.num_actions = num_actions

        # player index used for live graphing
        self.player_index = player_index

        # for interfacing with the rest of the world
        self.action_handler = action_handler

        super().__init__(session,
                         input_dim=self.state_dim,
                         output_dim=self.num_actions,
                         is_training=is_training,
                         optimizer=optimizer,
                         summary_writer=summary_writer,
                         summary_every=summary_every,
                         config_file=config_file)

    def create_input_array(self, game_tick_packet, frame_time):
        """Creates the input array from the game_tick_packet"""
        return self.input_formatter.create_input_array(game_tick_packet, frame_time)

    def apply_feature_creation(self, feature_creator):
        self.state_feature_dim = self.state_dim + tensorflow_feature_creator.get_feature_dim()
        self.feature_creator = feature_creator

    def sample_action(self, input_state):
        """
        Runs the model to get a single action that can be returned.
        :param input_state: This is the current state of the model at this point in time.
        :return:
        A sample action that can then be used to get controller output.
        """
        #always return an integer
        return self.sess.run(self.get_agent_output(), feed_dict={self.get_input_placeholder(): input_state})

    def get_agent_output(self):
        """
        :return: A tensor representing the output of the agent
        """
        return self.model
