import numpy as np
from rlbot.utils.structures import game_data_struct
from rlbot.utils.structures.game_data_struct import GameTickPacket

from examples.current.raw_input_formatter import RawInputFormatter


def get_state_dim():
    return 219


class LegacyGameInputFormatter(RawInputFormatter):
    last_total_score = 0

    """
    This is a class that takes in a game_tick_packet and will return an array of that value
    """

    def __init__(self, team, index):
        super().__init__()
        self.team = team
        self.index = index
        self.total_score = [0, 0]
        self.converted_array = [1] + self.get_input_state_dimension()

    def create_input_array(self, game_tick_packet: GameTickPacket, passed_time=0.0):
        """
        Creates an array for the model from the game_tick_packet
        :param game_tick_packet: A game packet for a single point in time
        :param passed_time: Time between the last frame and this one
        :return: A massive array representing that packet
        """

        if self.team == 1:
            game_data_struct.rotate_game_tick_packet_boost_omitted(game_tick_packet)

        player_car, team_members, enemies, own_team_score, enemy_team_score = self.split_teams(game_tick_packet)

        ball_data = self.get_ball_info(game_tick_packet)
        game_info = self.get_game_info(game_tick_packet)
        game_info.append(passed_time)

        boost_info = self.get_boost_info(game_tick_packet)

        score_info = self._get_score_info(game_tick_packet, enemy_team_score, own_team_score)
        # extra_features = feature_creator.get_extra_features(game_tick_packet, self.index)

        return self.create_result_array(game_info + score_info + player_car + ball_data +
                                        self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info)


    def _get_score_info(self, game_tick_packet, enemy_team_score, own_team_score):
        # we subtract so that when they score it becomes negative for this frame
        # and when we score it is positive
        total_score = enemy_team_score - own_team_score
        diff_in_score = self.last_total_score - total_score

        score_info = self.get_score_info(game_tick_packet.game_cars[self.index].score_info)
        score_info += [diff_in_score]

        self.last_total_score = total_score
        return score_info

    def split_teams(self, game_tick_packet: GameTickPacket):
        team_members = []
        enemies = []
        own_team_score = 0
        enemy_team_score = 0
        player_car = self.return_emtpy_player_array()
        for index in range(game_tick_packet.num_cars):
            if index == self.index:
                own_team_score += self.get_player_goals(game_tick_packet, index)
                enemy_team_score += self.get_own_goals(game_tick_packet, index)
                player_car = self.get_car_info(game_tick_packet, index)
            elif game_tick_packet.game_cars[index].team == self.team:
                own_team_score += self.get_player_goals(game_tick_packet, index)
                enemy_team_score += self.get_own_goals(game_tick_packet, index)
                team_members.append(self.get_car_info(game_tick_packet, index))
            else:
                enemies.append(self.get_car_info(game_tick_packet, index))
                enemy_team_score += self.get_player_goals(game_tick_packet, index)
                own_team_score += self.get_own_goals(game_tick_packet, index)
        while len(team_members) < 2:
            team_members.append(self.return_emtpy_player_array())
        while len(enemies) < 3:
            enemies.append(self.return_emtpy_player_array())
        return player_car, team_members, enemies, own_team_score, enemy_team_score

    def create_result_array(self, array):
        np_version = np.asarray(array, dtype=np.float32)
        output = np.argwhere(np.isnan(np_version))
        if len(output) > 0:
            print('nan indexes', output)
            for index in output:
                np_version[index[0]] = 0

        return np_version.reshape(self.converted_array)

    def get_player_goals(self, game_tick_packet: GameTickPacket, index):
        return game_tick_packet.game_cars[index].score_info.goals

    def get_own_goals(self, game_tick_packet: GameTickPacket, index):
        return game_tick_packet.game_cars[index].score_info.own_goals

    def return_emtpy_player_array(self):
        """
        :return: An array representing a car with no data
        """
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def flattenArrays(self, array_of_array):
        """
        Takes an array of arrays and flattens it into a single array
        :param array_of_array: A list that also contains a list
        :return: A single flattened array
        """
        return [item for sublist in array_of_array for item in sublist]

    def get_input_state_dimension(self):
        return [get_state_dim()]

    def get_ball_info(self, game_tick_packet: GameTickPacket):
        arr = super().get_ball_info(game_tick_packet)
        return arr[:11] + [0, 0, 0] + arr[11:]

    def get_car_info(self, game_tick_packet: GameTickPacket, index: int):
        arr = super().get_car_info(game_tick_packet, index)
        return arr[:-2] + [game_tick_packet.game_cars[index].team] + arr[-2:]
