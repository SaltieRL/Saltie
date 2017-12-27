import math
import tensorflow as tf


def get_output_vector(values, given_output):
    def distance(x1, y1, x2, y2):
        return tf.sqrt(tf.square(x2 - x1) + tf.square(y2 - y1))

    def to_degrees(radians):
        return radians * 180 / math.pi

    def aim(target_x, target_y):
        angle_between_bot_and_target = math.degrees(math.atan2(target_y - bot_pos.Y,
                                                               target_x - bot_pos.X))
        to_degrees(tf.atan2(tf.subtract(target_y, bot_pos.Y), tf.subtract(target_x, bot_pos.X)))
        angle_front_to_target = angle_between_bot_and_target - bot_yaw

        # Correct the values
        angle_front_to_target += tf.cond(tf.less(angle_front_to_target, -180), lambda: 360,
                                         lambda: tf.cond(tf.greater(angle_front_to_target, 180), lambda: -360,
                                                         lambda: 0))

        st = tf.cond(tf.less(angle_front_to_target, -10), lambda: -1,
                     lambda: tf.cond(tf.greater(angle_front_to_target, 10), lambda: 1, lambda: 0))

        ps = tf.cond(tf.less(tf.abs(to_degrees(angle_front_to_target)), powerslide_angle), lambda: True,
                     lambda: False)
        return [st, ps]

    index = 0

    # Contants
    distance_from_ball_to_boost = tf.constant(1500)  # Minimum distance to ball for using boost
    powerslide_angle = tf.constant(170)  # The angle (from the front of the bot to the ball) to start to powerslide.

    # Controller inputs
    # throttle = 0 defined in 100
    steer = 0
    pitch = 0
    yaw = 0
    roll = 0
    # boost = FalseThey
    jump = False
    powerslide = tf.constant(False)

    # Game values
    bot_pos = None
    bot_rot = None
    ball_pos = None
    bot_yaw = None

    # Update game data variables
    bot_pos = values.gamecars[index].Location
    bot_rot = values.gamecars[index].Rotation
    ball_pos = values.gameball.Location

    # Get car's yaw and convert from Unreal Rotator units to degrees
    bot_yaw = tf.abs(bot_rot.Yaw) % 65536 / 65536 * 360
    bot_yaw *= tf.cond(tf.less(bot_rot.Yaw, 0), lambda: -1, lambda: 1)

    # Boost when ball is far enough away
    boost = tf.cond(tf.greater(distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y), distance_from_ball_to_boost),
                    lambda: True, lambda: False)

    # Blue has their goal at -5000 (Y axis) and orange has their goal at 5000 (Y axis). This means that:
    # - Blue is behind the ball if the ball's Y axis is greater than blue's Y axis
    # - Orange is behind the ball if the ball's Y axis is smaller than orange's Y axis
    throttle = 1

    [steer, powerslide] = tf.cond(tf.logical_or(tf.logical_and(tf.equal(index, 0), tf.less(bot_pos.Y, ball_pos.Y)),
                                                tf.logical_and(tf.equal(index, 1), tf.greater(bot_pos.Y, ball_pos.Y))),
                                  lambda: aim(ball_pos.X, ball_pos.Y),
                                  lambda: tf.cond(tf.equal(index, 0), lambda: aim(0, -5000), lambda: aim(0, 5000)))

    # Boost on kickoff

    boost = tf.cond(tf.logical_and(tf.equal(ball_pos.X, 0), tf.equal(ball_pos.Y, 0)), lambda: 1, lambda: boost)
    throttle = tf.cond(tf.logical_and(tf.equal(ball_pos.X, 0), tf.equal(ball_pos.Y, 0)), lambda: 1, lambda: throttle)
    [steer, powerslide] = tf.cond(tf.logical_and(tf.equal(ball_pos.X, 0), tf.equal(ball_pos.Y, 0)),
                                  lambda: aim(ball_pos.X, ball_pos.Y), lambda: [steer, powerslide])

    def output_on_ground():
        # Throttle
        output = tf.cond(tf.less(tf.abs(throttle - given_output[0]), tf.constant(0.5)), lambda: 1, lambda: -1)

        # Steer
        output += tf.cond(tf.less_equal(tf.abs(steer - given_output[1]), tf.constant(0.5)), lambda: 1, lambda: -1)

        # Powerslide
        output += tf.cond(tf.equal(powerslide, given_output[7]), lambda: 1, lambda: -1)
        return output

    def output_off_ground():
        # Pitch
        output += tf.cond(tf.equal(pitch, given_output[2]), lambda: 1, lambda: -1)
        return output

    output = tf.cond(values.gamecars[index].bOnGround, output_on_ground(), output_off_ground())

    # Jump
    output += tf.cond(tf.equal(jump, given_output[5]), lambda: 1, lambda: -1)

    # Boost
    output += tf.cond(tf.equal(boost, given_output[6]), lambda: 1, lambda: -1)
    return output
