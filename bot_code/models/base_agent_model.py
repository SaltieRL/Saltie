import tensorflow as tf


from bot_code.models.base_model import BaseModel
from bot_code.conversions.input.input_formatter import InputFormatter

class BaseAgentModel(BaseModel):
    '''
    A base class for any model that outputs car actions given the the current state (+features).
    '''
    def __init__(self,
                 session,
                 num_actions,
                 input_formatter_info=[0, 0],
                 player_index=-1,
                 action_handler=None,
                 **kwargs):
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

        super().__init__(session, input_dim=self.state_dim, output_dim=num_actions, **kwargs)

    def create_input_array(self, game_tick_packet, frame_time):
        """Creates the input array from the game_tick_packet"""
        return self.input_formatter.create_input_array(game_tick_packet, frame_time)

    def apply_feature_creation(self, feature_creator):
        self.state_feature_dim = self.state_dim + tensorflow_feature_creator.get_feature_dim()
        self.feature_creator = feature_creator

