from math import sqrt
from bot_code.conversions import output_formatter
import math

UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180


def get_extra_features_from_array(array):
    car_loc = output_formatter.GAME_INFO_OFFSET + output_formatter.SCORE_INFO_OFFSET
    car_x = array[car_loc]
    car_y = array[car_loc + 1]
    car_z = array[car_loc + 2]
    car_rot_pitch = array[car_loc + 3]
    car_rot_yaw = array[car_loc + 4]

    ball_loc = car_loc + output_formatter.CAR_INFO_OFFSET
    ball_x = array[ball_loc]
    ball_y = array[ball_loc + 1]
    ball_z = array[ball_loc + 2]

    features = []
    features += generate_angles(car_x, car_y, car_z, ball_x, ball_y, ball_z, car_rot_pitch, car_rot_yaw)
    features.append(get_distance(car_x, car_y, car_z, ball_x, ball_y, ball_z))
    return features


def get_extra_features(game_tick_packet, self_index):
    car = game_tick_packet.gamecars[self_index]
    ball = game_tick_packet.gameball
    angles = generate_angles_loc(car, ball)
    distance = [get_distance_location(car.Location, ball.Location)]
    return angles + distance


def get_distance(x1, y1, z1, x2, y2, z2):
    # print(x1, y1, z1, x2, y2, z2)
    return sqrt((x1 - x2)**2 +
                (y1 - y2)**2 +
                (z1 - z2)**2)


def get_distance_location(location1, location2):
    return sqrt((location1.X - location2.X)**2 +
                (location1.Y - location2.Y)**2 +
                (location1.Z - location2.Z)**2)


def generate_angles(player_x, player_y, player_z, ball_x, ball_y, ball_z, pitch, yaw):

    # Nose vector x component
    player_rot1 = math.cos(pitch * URotationToRadians) * math.cos(yaw * URotationToRadians)
    # Nose vector y component
    player_rot4 = math.cos(pitch * URotationToRadians) * math.sin(yaw * URotationToRadians)
    # Nose vector z component
    # player_rot2 = math.sin(pitch * URotationToRadians)

    # Need to handle atan2(0,0) case, aka straight up or down, eventually
    player_front_direction_in_radians = math.atan2(player_rot1, player_rot4)
    relative_angle_to_ball_in_radians = math.atan2((ball_x - player_x), (ball_y - player_y))

    #player_front_direction_in_radians_XZ = math.atan2(player_rot1, player_rot2)
    #relative_angle_to_ball_in_radians_XZ = math.atan2((ball_x - player_x), (ball_z - player_z))

    return [player_front_direction_in_radians - relative_angle_to_ball_in_radians]


def generate_angles_loc(car, ball):
    return generate_angles(car.Location.X, car.Location.Y, car.Location.Z,
                           ball.Location.X, ball.Location.Y, ball.Location.Z,
                           float(car.Rotation.Pitch), float(car.Rotation.Yaw))
