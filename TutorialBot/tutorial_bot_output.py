import math
import tensorflow as tf


class TutorialBotOutput:
    # Constants
    distance_from_ball_to_boost = tf.constant(1500.0)  # Minimum distance to ball for using boost
    unreal_to_degrees = tf.constant(
        1.0 / 65536.0 * 360.0)  # The numbers used to convert unreal rotation units to degrees
    true = tf.constant(1.0)
    one = true
    false = tf.constant(0.0)
    zero = false

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def distance(self, x1, y1, x2, y2):
        return tf.sqrt(tf.square(x2 - x1) + tf.square(y2 - y1))

    def to_degrees(self, radians):
        return radians * 180 / math.pi

    def aim(self, bot_X, bot_Y, bot_Z, bot_yaw, target_x, target_y, target_z, is_on_ground):
        angle_between_bot_and_target = self.to_degrees(
            tf.atan2(tf.subtract(target_y, bot_Y), tf.subtract(target_x, bot_X)))
        angle_front_to_target = angle_between_bot_and_target - bot_yaw

        angle_front_to_target += (tf.cast(tf.less(angle_front_to_target, -180.0), tf.float32) * 360.0 +
                                  tf.cast(tf.less(angle_front_to_target, 180.0), tf.float32) * -360.0)

        full_turn_angle = 80
        half_turn_angle = 40
        powerslide_angle = tf.constant(160.0)  # The angle (from the front of the bot to the ball) to start to powerslide.
        absolute_angle = tf.abs(angle_front_to_target)

        # if between half_turn_angle and full_turn_angle
        half_turn = tf.logical_and(tf.greater_equal(absolute_angle, half_turn_angle),
                                   tf.less(full_turn_angle, full_turn_angle))

        half_turn_mult = 1.0 - tf.cast(half_turn, tf.float32) * 0.5

        turn_left = tf.cast(tf.less(angle_front_to_target, -half_turn_mult), tf.float32) # if angle < -full_turn_angle
        turn_right = tf.cast(tf.greater(angle_front_to_target, half_turn_mult), tf.float32) # if angle > full_turn_angle

        steer = - turn_left * half_turn_mult + turn_right * half_turn_mult

        vertical_distance = target_z - bot_Z
        should_jump = tf.logical_and(tf.greater(vertical_distance, 100), is_on_ground)

        jump = tf.cast(should_jump, tf.float32)

        ps = tf.greater(tf.abs(angle_front_to_target), powerslide_angle)
        power_slide = tf.cast(ps, tf.float32)
        return (steer, power_slide, jump)

    def get_output_vector(self, values):
        # Controller inputs
        # throttle = 0 defined in 100
        steer = tf.constant([0.0] * self.batch_size)
        pitch = tf.constant([0.0] * self.batch_size)
        yaw = tf.constant([0.0] * self.batch_size)
        roll = tf.constant([0.0] * self.batch_size)
        throttle = tf.constant([1.0] * self.batch_size)
        # boost = FalseThey
        jump = tf.constant([0.0] * self.batch_size)
        powerslide = tf.constant([0.0] * self.batch_size)

        # Update game data variables
        bot_pos = values.gamecars[0].Location
        bot_rot = values.gamecars[0].Rotation
        ball_pos = values.gameball.Location
        is_on_ground = values.gamecars[0].bOnGround
        car_boost = values.gamecars[0].Boost

        # Get car's yaw and convert from Unreal Rotator units to degrees
        bot_yaw = (tf.abs(bot_rot.Yaw) % 65536.0) * self.unreal_to_degrees
        # multiple by sign or raw data
        bot_yaw *= tf.sign(bot_rot.Yaw)
        xy_distance = self.distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y)

        # Boost when ball is far enough away
        boost = tf.logical_and(tf.greater(xy_distance, self.distance_from_ball_to_boost),
                               tf.greater(car_boost, 34))

        throttle = tf.cast(is_on_ground, tf.float32)
        blue_goal = tf.constant(-5000.0)
        go_to_ball = tf.cast(tf.less(bot_pos.Y, ball_pos.Y), tf.float32)
        go_to_goal = 1 - go_to_ball
        target_x = ball_pos.X * go_to_ball + self.zero * go_to_goal
        target_y = ball_pos.Y * go_to_ball + blue_goal * go_to_goal
        target_z = ball_pos.Z * go_to_ball + self.zero * go_to_goal

        steer, powerslide, jump = self.aim(bot_pos.X, bot_pos.Y, bot_pos.Z, bot_yaw,
                                           target_x, target_y, target_z, is_on_ground)

        # Boost on kickoff
        output = [throttle, steer, pitch, yaw, roll, jump, tf.cast(boost, tf.float32), powerslide]
        return output
