"""
Functions to convert from a state_array (see input_formatter)
to tensorflow objects similar to a game_tick_packet.
"""


# the length each of these takes in the state_array
GAME_INFO_OFFSET = 2
SCORE_INFO_OFFSET = 8
CAR_INFO_OFFSET = 20
BALL_INFO_OFFSET = 21


def get_game_info_index():
    return 0


def get_score_info_index():
    return GAME_INFO_OFFSET


def get_car_info_index():
    return GAME_INFO_OFFSET + SCORE_INFO_OFFSET


def get_ball_info_index():
    return GAME_INFO_OFFSET + SCORE_INFO_OFFSET + CAR_INFO_OFFSET


def get_basic_state(state_array):
    score_info = get_score_info(state_array, GAME_INFO_OFFSET)
    car_location = create_3D_point(state_array, GAME_INFO_OFFSET + SCORE_INFO_OFFSET)

    ball_location = create_3D_point(state_array,
                                    GAME_INFO_OFFSET +
                                    SCORE_INFO_OFFSET +
                                    CAR_INFO_OFFSET)
    has_last_touched_ball = state_array[GAME_INFO_OFFSET +
                                  SCORE_INFO_OFFSET +
                                  CAR_INFO_OFFSET - 1]
    state_object = create_object()
    state_object.score_info = score_info
    state_object.car_location = car_location
    state_object.ball_location = ball_location
    state_object.has_last_touched_ball = has_last_touched_ball
    return state_object


def get_advanced_state(state_array):
    car_info = get_car_info(state_array, GAME_INFO_OFFSET + SCORE_INFO_OFFSET)
    ball_info = get_ball_info(state_array, GAME_INFO_OFFSET +
                              SCORE_INFO_OFFSET +
                              CAR_INFO_OFFSET)
    state_object = create_object()
    state_object.car_info = car_info
    state_object.ball_info = ball_info
    state_object.gamecars = [car_info]
    state_object.gameball = ball_info

    return state_object


def is_empty_player_state_array(state_array, index, offset):
    sublist = state_array[index:index + offset]
    return all(p == 0.0 for p in sublist)


def create_object():
    return lambda: None


def create_3D_point(state_array, index):
    point = create_object()
    point.X = state_array[index]
    point.Y = state_array[index + 1]
    point.Z = state_array[index + 2]
    return point


def create_3D_rotation(state_array, index):
    point = create_object()
    point.Pitch = state_array[index]
    point.Yaw = state_array[index + 1]
    point.Roll = state_array[index + 2]
    return point


def get_car_info(state_array, index):
    car_info = create_object()
    car_info.Location = create_3D_point(state_array, index)
    car_info.Rotation = create_3D_rotation(state_array, index + 3)
    car_info.Velocity = create_3D_point(state_array, index + 6)
    car_info.AngularVelocity = create_3D_point(state_array, index + 9)
    car_info.bOnGround = state_array[index + 12]
    car_info.bSuperSonic = state_array[index + 13]
    car_info.bDemolished = state_array[index + 14]
    car_info.bJumped = state_array[index + 15]
    car_info.bDoubleJumped = state_array[index + 16]
    car_info.Team = state_array[index + 17]
    car_info.Boost = state_array[index + 18]
    car_info.bLastTouchedBall = state_array[index + 19]
    return car_info


def get_game_info(state_array, index):
    game_info = create_object()
    game_info.bBallHasBeenHit = (state_array[index] == 1)

    # no need for any of these but ball has been hit (kickoff indicator)
    # game_timeseconds = gameTickPacket.gameInfo.TimeSeconds
    # game_timeremaining = gameTickPacket.gameInfo.GameTimeRemaining
    # game_overtime = gameTickPacket.gameInfo.bOverTime
    # game_active = gameTickPacket.gameInfo.bRoundActive
    # game_ended = gameTickPacket.gameInfo.bMatchEnded
    return game_info


def get_ball_info(state_array, index):
    ball_info = create_object()
    ball_info.Location = create_3D_point(state_array, index)
    ball_info.Rotation = create_3D_rotation(state_array, index + 3)
    ball_info.Velocity = create_3D_point(state_array, index + 6)
    ball_info.AngularVelocity = create_3D_point(state_array, index + 9)
    ball_info.Acceleration = create_3D_point(state_array, index + 12)
    return ball_info


def get_boost_info(state_array, index):
    boost_objects = []
    for i in range(index, len(state_array), 2):
        boost_info = create_object()
        boost_info.bActive = (state_array[i] == 1)
        boost_info.Timer = state_array[i + 1]
        boost_objects.append(boost_info)
    return boost_objects

def get_score_info(state_array, index):
    score_info = create_object()
    score_info.Score = state_array[index]
    score_info.Goals = state_array[index + 1]
    score_info.OwnGoals = state_array[index + 2]
    score_info.Assists = state_array[index + 3]
    score_info.Saves = state_array[index + 4]
    score_info.Shots = state_array[index + 5]
    score_info.Demolitions = state_array[index + 6]
    score_info.FrameScoreDiff = state_array[index + 7]
    return score_info


def get_game_tick_packet(player_index):
    return None
