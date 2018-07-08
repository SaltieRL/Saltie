from framework.input_formatter.base_input_formatter import BaseInputFormatter
from rlbot.utils.structures.game_data_struct import GameTickPacket, ScoreInfo
from rlbot.utils.structures.start_match_structures import MAX_PLAYERS
from numpy import array


class RawInputFormatter(BaseInputFormatter):
    def create_input_array(self, game_tick_packet: GameTickPacket):
        input_array = []

        for index in range(MAX_PLAYERS):
            car_array = self.get_car_info(game_tick_packet, index)
            input_array.extend(car_array)

        ball_array = self.get_ball_info(game_tick_packet)
        input_array.extend(ball_array)

        return array(input_array)

    def get_game_info(self, game_tick_packet: GameTickPacket):
        game_ball_hit = game_tick_packet.game_info.is_kickoff_pause

        # no need for any of these but ball has been hit (kickoff indicator)
        # game_timeseconds = game_tick_packet.gameInfo.TimeSeconds
        # game_timeremaining = game_tick_packet.gameInfo.GameTimeRemaining
        # game_overtime = game_tick_packet.gameInfo.bOverTime
        # game_active = game_tick_packet.gameInfo.bRoundActive
        # game_ended = game_tick_packet.gameInfo.bMatchEnded
        return [game_ball_hit]

    def get_boost_info(self, game_tick_packet: GameTickPacket):
        game_inputs = []
        # limit this to 33 boosts total
        for i in range(game_tick_packet.num_boost):
            game_inputs.append(game_tick_packet.game_boosts[i].is_active)
            game_inputs.append(game_tick_packet.game_boosts[i].timer)
        return game_inputs

    def get_score_info(self, score_info: ScoreInfo):
        score = score_info.score
        goals = score_info.goals
        own_goals = score_info.own_gGoals
        assists = score_info.assists
        saves = score_info.saves
        shots = score_info.shots
        demolitions = score_info.demolitions

        return [score, goals, own_goals, assists, saves, shots, demolitions]

    def get_input_state_dimension(self):
        total = 0
        # game info
        total += 1
        # car info
        total += 19 * MAX_PLAYERS
        # ball info
        total += 18

        return [total]

    def get_car_info(self, game_tick_packet: GameTickPacket, index: int):
        player_x = game_tick_packet.game_cars[index].physics.location.x
        player_y = game_tick_packet.game_cars[index].physics.location.y
        player_z = game_tick_packet.game_cars[index].physics.location.z
        player_pitch = game_tick_packet.game_cars[index].physics.rotation.pitch
        player_yaw = game_tick_packet.game_cars[index].physics.rotation.yaw
        player_roll = game_tick_packet.game_cars[index].physics.rotation.roll
        player_speed_x = game_tick_packet.game_cars[index].physics.velocity.x
        player_speed_y = game_tick_packet.game_cars[index].physics.velocity.y
        player_speed_z = game_tick_packet.game_cars[index].physics.velocity.z
        player_angular_speed_x = game_tick_packet.game_cars[index].physics.angular_velocity.x
        player_angular_speed_y = game_tick_packet.game_cars[index].physics.angular_velocity.y
        player_angular_speed_z = game_tick_packet.game_cars[index].physics.angular_velocity.z
        player_is_demolished = game_tick_packet.game_cars[index].is_demolished
        player_has_wheel_contact = game_tick_packet.game_cars[index].has_wheel_contact
        player_is_super_sonic = game_tick_packet.game_cars[index].is_super_sonic
        player_jumped = game_tick_packet.game_cars[index].jumped
        player_double_jumped = game_tick_packet.game_cars[index].double_jumped
        player_boost = game_tick_packet.game_cars[index].boost
        last_touched_ball = self.has_last_touched_ball(game_tick_packet, index)
        car_array = [player_x, player_y, player_z,
                     player_pitch, player_yaw, player_roll,
                     player_speed_x, player_speed_y, player_speed_z,
                     player_angular_speed_x, player_angular_speed_y, player_angular_speed_z,
                     player_has_wheel_contact,
                     player_is_super_sonic,
                     player_is_demolished,
                     player_jumped, player_double_jumped,
                     player_boost,
                     last_touched_ball]

        return car_array

    def has_last_touched_ball(self, game_tick_packet: GameTickPacket, index: int):
        return game_tick_packet.game_cars[index].name == game_tick_packet.game_ball.latest_touch.player_name

    def get_ball_info(self, game_tick_packet: GameTickPacket):
        ball_x = game_tick_packet.game_ball.physics.location.x
        ball_y = game_tick_packet.game_ball.physics.location.y
        ball_z = game_tick_packet.game_ball.physics.location.z
        ball_pitch = game_tick_packet.game_ball.physics.rotation.pitch
        ball_yaw = game_tick_packet.game_ball.physics.rotation.yaw
        ball_roll = game_tick_packet.game_ball.physics.rotation.roll
        ball_speed_x = game_tick_packet.game_ball.physics.velocity.x
        ball_speed_y = game_tick_packet.game_ball.physics.velocity.y
        ball_speed_z = game_tick_packet.game_ball.physics.velocity.z
        ball_angular_speed_x = game_tick_packet.game_ball.physics.angular_velocity.x
        ball_angular_speed_y = game_tick_packet.game_ball.physics.angular_velocity.y
        ball_angular_speed_z = game_tick_packet.game_ball.physics.angular_velocity.z

        # touch info
        ball_touch_x = game_tick_packet.game_ball.latest_touch.hit_location.x
        ball_touch_y = game_tick_packet.game_ball.latest_touch.hit_location.y
        ball_touch_z = game_tick_packet.game_ball.latest_touch.hit_location.z
        ball_touch_speed_x = game_tick_packet.game_ball.latest_touch.hit_normal.x
        ball_touch_speed_y = game_tick_packet.game_ball.latest_touch.hit_normal.y
        ball_touch_speed_z = game_tick_packet.game_ball.latest_touch.hit_normal.z
        ball_array = [ball_x, ball_y, ball_z,
                      ball_pitch, ball_yaw, ball_roll,
                      ball_speed_x, ball_speed_y, ball_speed_z,
                      ball_angular_speed_x, ball_angular_speed_y, ball_angular_speed_z,
                      ball_touch_x, ball_touch_y, ball_touch_z,
                      ball_touch_speed_x, ball_touch_speed_y, ball_touch_speed_z]

        return ball_array
