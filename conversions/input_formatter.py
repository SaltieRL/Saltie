import numpy as np
import game_data_struct


def get_state_dim_with_features():
    return 206


class InputFormatter:
    last_total_score = 0

    """
    This is a class that takes in a game_tick_packet and will return an array of that value
    """

    def __init__(self, team, index):
        self.team = team
        self.index = index
        self.total_score = [0, 0]

    def create_input_array(self, game_tick_packet):
        """

        :param game_tick_packet: A game packet for a single point in time
        :return: A massive array representing that packet
        """

        if self.team == 1:
            game_data_struct.rotate_game_tick_packet_boost_omitted(game_tick_packet)

        team_members = []
        enemies = []
        ownTeamScore = 0
        enemyTeamScore = 0
        player_car = self.return_emtpy_player_array()
        for index in range(game_tick_packet.numCars):
            if index == self.index:
                ownTeamScore += self.get_player_goals(game_tick_packet, index)
                enemyTeamScore += self.get_own_goals(game_tick_packet, index)
                player_car = self.get_car_info(game_tick_packet, index)
            elif game_tick_packet.gamecars[index].Team == self.team:
                ownTeamScore += self.get_player_goals(game_tick_packet, index)
                enemyTeamScore += self.get_own_goals(game_tick_packet, index)
                team_members.append(self.get_car_info(game_tick_packet, index))
            else:
                enemies.append(self.get_car_info(game_tick_packet, index))
                enemyTeamScore += self.get_player_goals(game_tick_packet, index)
                ownTeamScore += self.get_own_goals(game_tick_packet, index)
        while len(team_members) < 2:
            team_members.append(self.return_emtpy_player_array())
        while len(enemies) < 3:
            enemies.append(self.return_emtpy_player_array())

        ball_data = self.get_ball_info(game_tick_packet)
        game_info = self.get_game_info(game_tick_packet)
        boost_info = self.get_boost_info(game_tick_packet)
        score_info = self.get_score_info(game_tick_packet.gamecars[self.index].Score)
        total_score = enemyTeamScore - ownTeamScore
        # we subtract so that when they score it becomes negative for this frame
        # and when we score it is positive
        diff_in_score = self.last_total_score - total_score
        score_info.append(diff_in_score)
        self.last_total_score = total_score
        # extra_features = feature_creator.get_extra_features(game_tick_packet, self.index)

        return self.create_result_array(game_info + score_info + player_car + ball_data +
                        self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info), \
               []

    def create_result_array(self, array):
        return np.array(array, dtype=np.float32)

    def get_player_goals(self, game_tick_packet, index):
        return game_tick_packet.gamecars[index].Score.Goals

    def get_own_goals(self, game_tick_packet, index):
        return game_tick_packet.gamecars[index].Score.OwnGoals

    def return_emtpy_player_array(self):
        """
        :return: An array representing a car with no data
        """
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def get_car_info(self, game_tick_packet, index):
        player_x = game_tick_packet.gamecars[index].Location.X
        player_y = game_tick_packet.gamecars[index].Location.Y
        player_z = game_tick_packet.gamecars[index].Location.Z
        player_pitch = game_tick_packet.gamecars[index].Rotation.Pitch
        player_yaw = game_tick_packet.gamecars[index].Rotation.Yaw
        player_roll = game_tick_packet.gamecars[index].Rotation.Roll
        player_speed_x = game_tick_packet.gamecars[index].Velocity.X
        player_speed_y = game_tick_packet.gamecars[index].Velocity.Y
        player_speed_z = game_tick_packet.gamecars[index].Velocity.Z
        player_angular_speed_x = game_tick_packet.gamecars[index].AngularVelocity.X
        player_angular_speed_y = game_tick_packet.gamecars[index].AngularVelocity.Y
        player_angular_speed_z = game_tick_packet.gamecars[index].AngularVelocity.Z
        player_demolished = game_tick_packet.gamecars[index].bDemolished
        player_jumped = game_tick_packet.gamecars[index].bJumped
        player_double_jumped = game_tick_packet.gamecars[index].bDoubleJumped
        player_team = game_tick_packet.gamecars[index].Team
        player_boost = game_tick_packet.gamecars[index].Boost
        last_touched_ball = self.get_last_touched_ball(game_tick_packet.gamecars[index], game_tick_packet.gameball.LatestTouch)
        return [player_x, player_y, player_z, player_pitch, player_yaw, player_roll,
                player_speed_x, player_speed_y, player_speed_z, player_angular_speed_x,
                player_angular_speed_y, player_angular_speed_z, player_demolished, player_jumped,
                player_double_jumped, player_team, player_boost, last_touched_ball]

    def get_last_touched_ball(self, car, latest_touch):
        return (car.wName == latest_touch.wPlayerName)

    def get_game_info(self, game_tick_packet):
        game_ball_hit = game_tick_packet.gameInfo.bBallHasBeenHit

        # no need for any of these but ball has been hit (kickoff indicator)
        # game_timeseconds = game_tick_packet.gameInfo.TimeSeconds
        # game_timeremaining = game_tick_packet.gameInfo.GameTimeRemaining
        # game_overtime = game_tick_packet.gameInfo.bOverTime
        # game_active = game_tick_packet.gameInfo.bRoundActive
        # game_ended = game_tick_packet.gameInfo.bMatchEnded
        return [game_ball_hit]

    def get_ball_info(self, game_tick_packet):
        ball_x = game_tick_packet.gameball.Location.X
        ball_y = game_tick_packet.gameball.Location.Y
        ball_z = game_tick_packet.gameball.Location.Z
        ball_pitch = game_tick_packet.gameball.Rotation.Pitch
        ball_yaw = game_tick_packet.gameball.Rotation.Yaw
        ball_roll = game_tick_packet.gameball.Rotation.Roll
        ball_speed_x = game_tick_packet.gameball.Velocity.X
        ball_speed_y = game_tick_packet.gameball.Velocity.Y
        ball_speed_z = game_tick_packet.gameball.Velocity.Z
        ball_angular_speed_x = game_tick_packet.gameball.AngularVelocity.X
        ball_angular_speed_y = game_tick_packet.gameball.AngularVelocity.Y
        ball_angular_speed_z = game_tick_packet.gameball.AngularVelocity.Z
        ball_acceleration_x = game_tick_packet.gameball.Acceleration.X
        ball_acceleration_y = game_tick_packet.gameball.Acceleration.Y
        ball_acceleration_z = game_tick_packet.gameball.Acceleration.Z

        # touch info
        ball_touch_x = game_tick_packet.gameball.LatestTouch.sHitLocation.X
        ball_touch_y = game_tick_packet.gameball.LatestTouch.sHitLocation.Y
        ball_touch_z = game_tick_packet.gameball.LatestTouch.sHitLocation.Z
        ball_touch_speed_x = game_tick_packet.gameball.LatestTouch.sHitNormal.X
        ball_touch_speed_y = game_tick_packet.gameball.LatestTouch.sHitNormal.Y
        ball_touch_speed_z = game_tick_packet.gameball.LatestTouch.sHitNormal.Z
        return [ball_x, ball_y, ball_z,
                ball_pitch, ball_yaw, ball_roll,
                ball_speed_x, ball_speed_y, ball_speed_z,
                ball_angular_speed_x, ball_angular_speed_y, ball_angular_speed_z,
                ball_acceleration_x, ball_acceleration_y, ball_acceleration_z,
                ball_touch_x, ball_touch_y, ball_touch_z,
                ball_touch_speed_x, ball_touch_speed_y, ball_touch_speed_z]

    def get_boost_info(self, game_tick_packet):
        game_inputs = []
        # limit this to 33 boosts total
        for i in range(34):
            game_inputs.append(game_tick_packet.gameBoosts[i].bActive)
            game_inputs.append(game_tick_packet.gameBoosts[i].Timer)
        return game_inputs

    def get_score_info(self, Score):
        score = Score.Score
        goals = Score.Goals
        own_goals = Score.OwnGoals
        assists = Score.Assists
        saves = Score.Saves
        shots = Score.Shots
        demolitions = Score.Demolitions

        return [score, goals, own_goals, assists, saves, shots, demolitions]

    def flattenArrays(self, array_of_array):
        """
        Takes an array of arrays and flattens it into a single array
        :param array_of_array: A list that also contains a list
        :return: A single flattened array
        """
        return [item for sublist in array_of_array for item in sublist]

    def get_state_dim_with_features(self):
        return get_state_dim_with_features()
