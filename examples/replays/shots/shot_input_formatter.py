import pandas as pd
import numpy as np
from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.replay.replay_format import GeneratedHit


class ShotInputFormatter(BaseInputFormatter):
    def get_input_state_dimension(self):
        pass

    def create_input_array(self, input_data: GeneratedHit, batch_size=1):
        result = []
        hit = input_data.get_hit()
        frame = hit.frame_number
        df = input_data.get_replay().get_pandas()
        index = df.index
        hit_frame = df.loc[frame]

        new_frame = hit_frame['ball']

        proto = input_data.get_replay().get_proto()
        hit_player = hit.player_id
        blue_team = []
        orange_team = []
        for player in proto.players:
            if player.is_orange:
                orange_team.append(player)
            else:
                blue_team.append(player)

        return result

    def get_speed(self, frame):
        return np.sqrt(frame['vel_x']**2 + frame['vel_y']**2 + frame['vel_z']**2)

    def get_distance_from_goal(self, frame, player, team):
        if team == 0:
            return np.sqrt(frame['pos_x']**2 + (frame['pos_y'] - (6000 * (1 - team)))**2 + frame['vel_z']**2)

    def get_player_data(self, frame):
        return [frame['pos_x'], frame['pos_y'], frame['pos_z'],
                frame['rot_x'], frame['rot_y'], frame['rot_z'],
                frame['vel_x'], frame['vel_y'], frame['vel_z']]
