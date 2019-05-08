import math
import numpy as np

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick


class LeviInputFormatter(BaseInputFormatter):
    def __init__(self, team, index):
        super().__init__()
        self.team = team
        self.index = index

    def create_input_array(self, packet, batch_size=1):
        if batch_size != 1:
            raise NotImplementedError

        packet = packet[0]

        own_car = packet.game_cars[self.index]
        game_ball = packet.game_ball

        own_car_location = own_car.physics.location
        game_ball_location = game_ball.physics.location

        own_car_velocity = own_car.physics.velocity
        game_ball_velocity = game_ball.physics.velocity

        own_car_angular = own_car.physics.angular_velocity
        game_ball_angular = game_ball.physics.angular_velocity

        own_theta = get_all_vectors(own_car)

        spatial_x = np.array([own_car_location.x, game_ball_location.x,
                              own_car_velocity.x, game_ball_velocity.x,
                              own_car_angular.x, game_ball_angular.x])

        spatial_y = np.array([own_car_location.y, game_ball_location.y,
                              own_car_velocity.y, game_ball_velocity.y,
                              own_car_angular.y, game_ball_angular.y])

        spatial_z = np.array([own_car_location.z, game_ball_location.z,
                              own_car_velocity.z, game_ball_velocity.z,
                              own_car_angular.z, game_ball_angular.z])

        spatial = np.stack([spatial_x, spatial_y, spatial_z])
        spatial = np.concatenate([spatial, own_theta], axis=1)

        spatial[:, 0:4] /= 1000

        own_car_stats = np.array([own_car.boost / 100,
                                  1 if own_car.jumped else 0,
                                  1 if own_car.double_jumped else 0,
                                  1 if own_car.is_demolished else 0,
                                  1 if own_car.has_wheel_contact else 0])

        if self.team == 1:
            spatial[0:2] *= -1

        return [np.expand_dims(spatial, axis=0), np.expand_dims(own_car_stats, axis=0)]

    def get_input_from_rigid(self, rigid_body_tick: RigidBodyTick):
        own_car = rigid_body_tick.players[self.index]
        ball = rigid_body_tick.ball

        own_car_location = own_car.state.location
        game_ball_location = ball.state.location

        own_car_velocity = own_car.state.velocity
        game_ball_velocity = ball.state.velocity

        own_car_angular = own_car.state.angular_velocity
        game_ball_angular = ball.state.angular_velocity

        q = own_car.state.rotation
        q = (q.w, q.x, q.y, q.z)
        own_theta = np.array([
            qv_mult(q, (1, 0, 0)),
            qv_mult(q, (0, 1, 0)),
            qv_mult(q, (0, 0, 1)),
        ]).swapaxes(0, 1)

        spatial_x = np.array([own_car_location.x, game_ball_location.x,
                              own_car_velocity.x, game_ball_velocity.x,
                              own_car_angular.x, game_ball_angular.x])

        spatial_y = np.array([own_car_location.y, game_ball_location.y,
                              own_car_velocity.y, game_ball_velocity.y,
                              own_car_angular.y, game_ball_angular.y])

        spatial_z = np.array([own_car_location.z, game_ball_location.z,
                              own_car_velocity.z, game_ball_velocity.z,
                              own_car_angular.z, game_ball_angular.z])

        spatial = np.stack([spatial_x, spatial_y, spatial_z])
        spatial = np.concatenate([spatial, own_theta], axis=1)

        spatial[:, 0:4] /= 1000

        own_car_stats = np.zeros(5)

        if self.team == 1:
            spatial[0:2] *= -1

        return [np.expand_dims(spatial, axis=0), np.expand_dims(own_car_stats, axis=0)]

    def get_input_state_dimension(self):
        return [(3, 9), (5,)]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def q_conjugate(q):
    w, x, y, z = q
    return w, -x, -y, -z


def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


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
