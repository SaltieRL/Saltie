# the length each of these takes in the array
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


def get_basic_state(array):
    score_info = get_score_info(array, GAME_INFO_OFFSET)
    car_location = create_3D_point(array, GAME_INFO_OFFSET + SCORE_INFO_OFFSET)

    ball_location = create_3D_point(array,
                                    GAME_INFO_OFFSET +
                                    SCORE_INFO_OFFSET +
                                    CAR_INFO_OFFSET)
    has_last_touched_ball = array[GAME_INFO_OFFSET +
                                  SCORE_INFO_OFFSET +
                                  CAR_INFO_OFFSET - 1]
    result = create_object()
    result.score_info = score_info
    result.car_location = car_location
    result.ball_location = ball_location
    result.has_last_touched_ball = has_last_touched_ball
    return result


def get_advanced_state(input_array):
    car_info = get_car_info(input_array, GAME_INFO_OFFSET + SCORE_INFO_OFFSET)
    ball_info = get_ball_info(input_array, GAME_INFO_OFFSET +
                              SCORE_INFO_OFFSET +
                              CAR_INFO_OFFSET)
    result = create_object()
    result.car_info = car_info
    result.ball_info = ball_info

    return result


def is_empty_player_array(array, index, offset):
    sublist = array[index:index + offset]
    return all(p == 0.0 for p in sublist)


def create_object():
    return lambda: None


def create_3D_point(array, index):
    point = create_object()
    point.X = array[index]
    point.Y = array[index + 1]
    point.Z = array[index + 2]
    return point


def create_3D_rotation(array, index):
    point = create_object()
    point.Pitch = array[index]
    point.Yaw = array[index + 1]
    point.Roll = array[index + 2]
    return point


def get_car_info(array, index):
    if is_empty_player_array(array, index, CAR_INFO_OFFSET):
        return None
    car_info = create_object()
    car_info.Location = create_3D_point(array, index)
    car_info.Rotation = create_3D_rotation(array, index + 3)
    car_info.Velocity = create_3D_point(array, index + 6)
    car_info.AngularVelocity = create_3D_point(array, index + 9)
    car_info.bOnGround = array[12]
    car_info.bSuperSonic = array[13]
    car_info.bDemolished = array[14]
    car_info.bJumped = array[15]
    car_info.bDoubleJumped = array[16]
    car_info.Team = array[17]
    car_info.Boost = array[18]
    car_info.bLastTouchedBall = array[19]
    return car_info


def get_game_info(array, index):
    game_info = create_object()
    game_info.bBallHasBeenHit = (array[index] == 1)

    # no need for any of these but ball has been hit (kickoff indicator)
    # game_timeseconds = gameTickPacket.gameInfo.TimeSeconds
    # game_timeremaining = gameTickPacket.gameInfo.GameTimeRemaining
    # game_overtime = gameTickPacket.gameInfo.bOverTime
    # game_active = gameTickPacket.gameInfo.bRoundActive
    # game_ended = gameTickPacket.gameInfo.bMatchEnded
    return game_info


def get_ball_info(array, index):
    ball_info = create_object()
    ball_info.Location = create_3D_point(array, index)
    ball_info.Rotation = create_3D_rotation(array, index + 3)
    ball_info.Velocity = create_3D_point(array, index + 6)
    ball_info.AngularVelocity = create_3D_point(array, index + 9)
    ball_info.Acceleration = create_3D_point(array, index + 12)
    return ball_info


def get_boost_info(array, index):
    boost_objects = []
    for i in range(index, len(array), 2):
        boost_info = create_object()
        boost_info.bActive = (array[i] == 1)
        boost_info.Timer = array[i + 1]
        boost_objects.append(boost_info)
    return boost_objects

def get_score_info(array, index):
    score_info = create_object()
    score_info.Score = array[index]
    score_info.Goals = array[index + 1]
    score_info.OwnGoals = array[index + 2]
    score_info.Assists = array[index + 3]
    score_info.Saves = array[index + 4]
    score_info.Shots = array[index + 5]
    score_info.Demolitions = array[index + 6]
    score_info.FrameScoreDiff = array[index + 7]
    return score_info
