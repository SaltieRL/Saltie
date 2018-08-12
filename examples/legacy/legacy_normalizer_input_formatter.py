from rlbot.utils.structures.game_data_struct import GameTickPacket, ScoreInfo
import numpy as np

from examples.current.raw_input_formatter import RawInputFormatter
from examples.legacy.legacy_game_input_formatter import LegacyGameInputFormatter
from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.input_formatter.host_input_formatter import HostInputFormatter


class LegacyNormalizerInputFormatter(HostInputFormatter):
    game_tick_formatter = None
    boolean = [0.0, 1.0]

    class __Expando(object):
        pass

    def __init__(self, input_formatter: BaseInputFormatter):
        super().__init__(input_formatter)
        self.game_tick_formatter = NormalizedGameInputFormatter(0, 0)
        self.normalization_array = self.game_tick_formatter.create_input_array(self.get_normalized_game_tick())

    def create_input_array(self, input_array, batch_size=1):
        converted_input_array = self.input_formatter.create_input_array(input_array)

        min = self.normalization_array[0]
        max = self.normalization_array[1]

        diff = max - min

        result = (input_array - min) / diff
        return result

    def get_normalized_game_tick(self) -> GameTickPacket:
        state_object = self.create_object()
        # Game info
        state_object.game_info = self.get_game_info()
        # Score info

        # Player car info
        state_object.game_cars = []
        car_info = self.get_car_info()
        for i in range(6):
            state_object.game_cars.append(car_info)

        state_object.num_cars = len(state_object.game_cars)

        # Ball info
        state_object.game_ball = self.get_ball_info()

        state_object.game_boosts = self.get_boost_info()
        state_object.num_boost = len(state_object.game_boosts)
        return state_object


    # game_info + score_info + player_car + ball_data +
    # self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info
    def create_object(self) -> GameTickPacket:
        return self.__Expando()

    def get_game_info(self):
        info = self.create_object()
        # Game info
        info.is_overtime = self.boolean
        info.is_unlimited_time = self.boolean
        info.is_round_active = self.boolean
        info.is_kickoff_pause = self.boolean
        info.is_match_ended = self.boolean

        return info

    def create_3D_point(self, x, y, z):
        point = self.create_object()
        point.x = x
        point.y = y
        point.z = z
        return point

    def create_3D_rotation(self, pitch, yaw, roll):
        rotator = self.create_object()
        rotator.pitch = pitch
        rotator.yaw = yaw
        rotator.roll = roll
        return rotator

    def create_physics(self, input_velocity, input_angular):
        physics = self.create_object()
        physics.location = self.get_location()
        physics.rotation = self.create_3D_rotation([-16384, 16384],  # Pitch
                                           [-32768, 32768],  # Yaw
                                           [-32768, 32768])  # Roll
        print(physics.location)

        physics.velocity = self.create_3D_point(
            [-input_velocity, input_velocity],  # Velocity X
            [-input_velocity, input_velocity],  # Y
            [-input_velocity, input_velocity])  # Z

        physics.angular_velocity = self.create_3D_point(
            [-input_angular, input_angular],  # Angular velocity X
            [-input_angular, input_angular],  # Y
            [-input_angular, input_angular])  # Z

        return physics

    def get_location(self):
        return self.create_3D_point(
            [-8300, 8300],  # Location X
            [-11800, 11800],  # Y
            [0, 2000])

    def get_car_info(self):
        car = self.create_object()

        car.physics = self.create_physics(2300, 5.5)

        car.is_demolished = self.boolean  # Demolished

        car.has_wheel_contact = self.boolean

        car.jumped = self.boolean  # Jumped
        car.is_super_sonic = self.boolean # Jumped

        car.double_jumped = self.boolean

        car.team = self.boolean

        car.boost = [0.0, 1]

        car.score_info = self.get_car_score_info()

        return car

    def get_car_score_info(self):
        score = self.create_object()
        score.score = [0, 100]
        score.goals = self.boolean
        score.own_goals = self.boolean
        score.assists = self.boolean
        score.saves = self.boolean
        score.shots = self.boolean
        score.demolitions = self.boolean
        return score

    def get_ball_info(self):
        ball = self.create_object()

        ball.physics = self.create_physics(6000.0, 6.0)

        ball.latest_touch = self.create_object()

        ball.latest_touch.hit_location = self.get_location()
        ball.latest_touch.hit_normal = ball.physics.velocity
        return ball

    def get_boost_info(self):
        boost_objects = []
        for i in range(34):
            boost_info = self.create_object()
            boost_info.is_active = self.boolean
            boost_info.timer = [0.0, 10000.0]
            boost_objects.append(boost_info)
        return boost_objects


class NormalizedGameInputFormatter(LegacyGameInputFormatter):
    def __init__(self, team, index):
        super().__init__(team, index)
        self.converted_array = self.get_input_state_dimension() + [2]

    def split_teams(self, game_tick_packet):
        team_members = []
        enemies = []
        own_team_score = [0, 10]
        enemy_team_score = [0, 10]
        player_car = self.get_car_info(game_tick_packet, 0)
        while len(team_members) < 2:
            team_members.append(player_car)
        while len(enemies) < 3:
            enemies.append(player_car)

        return player_car, team_members, enemies, own_team_score, enemy_team_score

    def has_last_touched_ball(self, car, latest_touch):
        return [0.0, 1.0]

    def get_score_info(self, score_info: ScoreInfo):
        result = super().get_score_info(score_info)

        # the change in score can only be -1 to 1
        result[-1] = [-1, 1]
        return result

    def create_input_array(self, game_tick_packet, passed_time=None):
        if passed_time is not None:
            return super().create_input_array(game_tick_packet, passed_time)
        return super().create_input_array(game_tick_packet, [0.0, 1.0])

    def get_ball_info(self, game_tick_packet: GameTickPacket):
        arr = RawInputFormatter.get_ball_info(self, game_tick_packet)
        return arr[:11] + [[0, 1], [0, 1], [0, 1]] + arr[11:]

    def _get_score_info(self, game_tick_packet, enemy_team_score, own_team_score):
        score_info = self.get_score_info(game_tick_packet.game_cars[self.index].score_info)
        score_info += [[0, 20]]

        return score_info

    def create_result_array(self, array):
        np_version = np.asarray(array, dtype=np.float32)
        output = np.argwhere(np.isnan(np_version))
        if len(output) > 0:
            print('nan indexes', output)
            for index in output:
                np_version[index[0]] = 0

        return np.swapaxes(np_version.reshape(self.converted_array), 0, 1)
