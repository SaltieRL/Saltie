import math
import numpy as np

from framework.input_formatter.base_input_formatter import BaseInputFormatter


class LeviInputFormatter(BaseInputFormatter):
    def __init__(self, team, index):
        super().__init__()
        self.team = team
        self.index = index

    def create_input_array(self, packet_list, batch_size=1):
        input_list = [self.create_input(packet_list[i]) for i in range(batch_size)]
        input_array = [np.stack([input_list[j][i] for j in range(batch_size)]) for i in range(7)]

        return input_array

    def create_input(self, packet):
        own_car_stats = None
        own_team_car_stats = np.empty((0, 6))  # player, value
        opp_team_car_stars = np.empty((0, 6))  # player, value

        own_car_spatial = None
        own_team_car_spatial = np.empty((0, 3, 6))  # player, axis, value
        opp_team_car_spatial = np.empty((0, 3, 6))  # player, axis, value

        for car_index in range(packet.num_cars):
            car = packet.game_cars[car_index]

            car_stats = np.array([car.boost / 100,
                                  1 if car.jumped else 0,
                                  1 if car.double_jumped else 0,
                                  1 if car.is_demolished else 0,
                                  1 if car.has_wheel_contact else 0,
                                  1 if car.is_supersonic else 0])

            location = car.physics.location
            velocity = car.physics.velocity
            angular = car.physics.angular_velocity

            car_spatial = np.array([[location.x, velocity.x, angular.x],
                                    [location.y, velocity.y, angular.y],
                                    [location.z, velocity.z, angular.z]])
            car_spatial[:, 0:2] /= 1000
            theta = get_all_vectors(car)

            car_spatial = np.concatenate((car_spatial, theta), axis=1)

            if self.team == 1:
                car_spatial[0:2] *= -1  # rotate the whole field

            if car_index == self.index:
                own_car_stats = car_stats
                own_car_spatial = car_spatial
            elif car.team == self.team:
                own_team_car_stats = np.concatenate((own_team_car_stats, [car_stats]))
                own_team_car_spatial = np.concatenate((own_team_car_spatial, [car_spatial]))
            else:
                opp_team_car_stars = np.concatenate((opp_team_car_stars, [car_stats]))
                opp_team_car_spatial = np.concatenate((opp_team_car_spatial, [car_spatial]))

        location = packet.game_ball.physics.location
        velocity = packet.game_ball.physics.velocity
        angular = packet.game_ball.physics.angular_velocity

        game_ball_spatial = np.array([[location.x, velocity.x, angular.x],
                                      [location.y, velocity.y, angular.y],
                                      [location.z, velocity.z, angular.z]])

        if self.team == 1:
            game_ball_spatial[0:2] *= -1  # rotate the whole field

        return [own_car_stats,
                own_team_car_stats,
                opp_team_car_stars,
                own_car_spatial,
                own_team_car_spatial,
                opp_team_car_spatial,
                game_ball_spatial]

    def get_input_state_dimension(self):
        return [(6,), (None, 6), (None, 6), (3, 6), (None, 3, 6), (None, 3, 6), (3, 3)]


def get_all_vectors(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)
    roll = float(car.physics.rotation.roll)

    c_r = math.cos(roll)
    s_r = math.sin(roll)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)
    c_y = math.cos(yaw)
    s_y = math.sin(yaw)

    theta = np.zeros((3, 3))
    #   front direction
    theta[0, 0] = c_p * c_y
    theta[1, 0] = c_p * s_y
    theta[2, 0] = s_p

    #   left direction
    theta[0, 1] = c_y * s_p * s_r - c_r * s_y
    theta[1, 1] = s_y * s_p * s_r + c_r * c_y
    theta[2, 1] = -c_p * s_r

    #   up direction
    theta[0, 2] = -c_r * c_y * s_p - s_r * s_y
    theta[1, 2] = -c_r * s_y * s_p + s_r * c_y
    theta[2, 2] = c_p * c_r

    return theta
