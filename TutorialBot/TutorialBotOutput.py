import math
import tensorflow as tf


def get_output_vector(values, given_output):
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def aim(target_x, target_y):
        angle_between_bot_and_target = math.degrees(math.atan2(target_y - bot_pos.Y,
                                                               target_x - bot_pos.X))

        angle_front_to_target = angle_between_bot_and_target - bot_yaw

        # Correct the values
        if angle_front_to_target < -180:
            angle_front_to_target += 360
        if angle_front_to_target > 180:
            angle_front_to_target -= 360

        nonlocal steer
        nonlocal powerslide

        if angle_front_to_target < -10:
            # If the target is more than 10 degrees right from the centre, steer left
            steer = -1
        elif angle_front_to_target > 10:
            # If the target is more than 10 degrees left from the centre, steer right
            steer = 1
        else:
            # If the target is less than 10 degrees from the centre, steer straight
            steer = 0

        if abs(math.degrees(angle_front_to_target)) < POWERSLIDE_ANGLE:
            powerslide = True
        else:
            powerslide = False

    # def check_for_dodge(target_x, target_y):
    #     if should_dodge and time.time() > next_dodge_time:
    #         aim(target_x, target_y)
    #         jump = True
    #         pitch = -1
    #
    #         if on_second_jump:
    #             on_second_jump = False
    #             should_dodge = False
    #         else:
    #             on_second_jump = True
    #             next_dodge_time = time.time() + DODGE_TIME

    index = 0

    # Contants
    DODGE_TIME = 0.2
    DISTANCE_TO_DODGE = 500
    DISTANCE_FROM_BALL_TO_BOOST = 1500  # The minimum distance the ball needs to be away from the bot for the bot to boost
    POWERSLIDE_ANGLE = 170  # The angle (from the front of the bot to the ball) at which the bot should start to powerslide.

    # Controller inputs
    # throttle = 0 defined in 100
    steer = 0
    pitch = 0
    yaw = 0
    roll = 0
    # boost = False
    jump = False
    powerslide = False

    # Game values
    bot_pos = None
    bot_rot = None
    ball_pos = None
    bot_yaw = None

    # Dodging
    # should_dodge = False
    # on_second_jump = False
    # next_dodge_time = 0

    # Update game data variables
    bot_pos = values.gamecars[index].Location
    bot_rot = values.gamecars[index].Rotation
    ball_pos = values.gameball.Location

    # Get car's yaw and convert from Unreal Rotator units to degrees
    bot_yaw = abs(bot_rot.Yaw) % 65536 / 65536 * 360
    if bot_rot.Yaw < 0:
        bot_yaw *= -1

    # Boost when ball is far enough away
    if distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y) > DISTANCE_FROM_BALL_TO_BOOST:
        boost = True
    else:
        boost = False

    # Blue has their goal at -5000 (Y axis) and orange has their goal at 5000 (Y axis). This means that:
    # - Blue is behind the ball if the ball's Y axis is greater than blue's Y axis
    # - Orange is behind the ball if the ball's Y axis is smaller than orange's Y axis
    throttle = 1
    if (index == 0 and bot_pos.Y < ball_pos.Y) or (index == 1 and bot_pos.Y > ball_pos.Y):
        aim(ball_pos.X, ball_pos.Y)

        # if distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y) < DISTANCE_TO_DODGE:
        #     should_dodge = True
    else:
        if index == 0:
            # Blue team's goal is located at (0, -5000)
            aim(0, -5000)
        else:
            # Orange team's goal is located at (0, 5000)
            aim(0, 5000)

    # Boost on kickoff
    if ball_pos.X == 0 and ball_pos.Y == 0:
        aim(ball_pos.X, ball_pos.Y)
        boost = True
        throttle = 1

    # check_for_dodge(ball_pos.X, ball_pos.Y)

    # output = [0] * 8  # Initialised all to neutral
    output = tf.Variable(0)
    if values.gamecars[index].bOnGround:
        # Throttle
        output += tf.cond(tf.less(tf.abs(throttle - given_output[0]), tf.constant(0.5)), lambda: 1, lambda: -1)

        # Steer
        output += tf.cond(tf.less_equal(tf.abs(steer - given_output[1]), tf.constant(0.5)), lambda: 1, lambda: -1)

        # Powerslide
        output += tf.cond(tf.equal(powerslide, given_output[7]), lambda: 1, lambda: -1)

    else:
        # Pitch
        output += tf.cond(tf.equal(pitch, given_output[2]), lambda: 1, lambda: -1)

    # Jump
    output += tf.cond(tf.equal(jump, given_output[5]), lambda: 1, lambda: -1)

    # Boost
    output += tf.cond(tf.equal(boost, given_output[6]), lambda: 1, lambda: -1)
    return output
