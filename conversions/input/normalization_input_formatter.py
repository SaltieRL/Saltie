import tensorflow as tf
from conversions.input import tensorflow_input_formatter


class NormalizationInputFormatter(tensorflow_input_formatter.TensorflowInputFormatter):
    def __init__(self, team, index, batch_size):
        super().__init__(team, index, batch_size)

    def split_teams(self, game_tick_packet):
        team_members = []
        enemies = []
        own_team_score = 0
        enemy_team_score = 0
        player_car = self.get_car_info(game_tick_packet, 0)
        while len(team_members) < 2:
            team_members.append(player_car)
        while len(enemies) < 3:
            enemies.append(player_car)

        return player_car, team_members, enemies, own_team_score, enemy_team_score

    def get_last_touched_ball(self, car, latest_touch):
        return tf.constant([0.0, 1.0])

    def create_result_array(self, array):
        converted_array = []
        for i in range(len(array)):
            casted_number = tf.cast(array[i], tf.float32)
            converted_array.append(casted_number)
        result = tf.stack(converted_array, axis=1)
        return result

    def get_score_info(self, score, diff_in_score):
        result = super().get_score_info(score, diff_in_score)

        # the change in score can only be -1 to 1
        result[len(result) - 1] = [-1, 1]
        return result

    def create_input_array(self, game_tick_packet, passed_time=None):
        if passed_time is not None:
            return super().create_input_array(game_tick_packet, passed_time)
        return super().create_input_array(game_tick_packet, tf.constant([0.0, 1.0]))
