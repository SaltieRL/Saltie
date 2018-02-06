import tensorflow as tf
from bot_code.modelHelpers import tensorflow_feature_creator

class TutorialBotOutput:
    # Constants
    distance_from_ball_to_go_fast = tf.constant(600.0)
    distance_from_ball_to_boost = tf.constant(2000.0)  # Minimum distance to ball for using boost
    unreal_to_degrees = tf.constant(
        1.0 / 65536.0 * 360.0)  # The numbers used to convert unreal rotation units to degrees
    true = tf.constant(1.0)
    one = true
    false = tf.constant(0.0)
    zero = false
    feature_creator = tensorflow_feature_creator.TensorflowFeatureCreator()

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def distance(self, x1, y1, x2, y2):
        return tf.sqrt(tf.square(x2 - x1) + tf.square(y2 - y1))

    def aim(self, bot_position, bot_rotation, target_x, target_y, target_z, distance_to_ball, is_on_ground):
        full_turn_angle = 70.0
        half_turn_angle = 30.0
        powerslide_angle_constant = 80.0 # The angle (from the front of the bot to the ball) to start to powerslide.

        angle_front_to_target = self.feature_creator.generate_angle_to_target(bot_position.X, bot_position.Y,
                                                                              bot_rotation,
                                                                              target_x, target_y)

        absolute_angle = tf.abs(angle_front_to_target)

        # if between half_turn_angle and full_turn_angle
        half_turn = tf.logical_and(tf.greater_equal(absolute_angle, half_turn_angle),
                                   tf.less(absolute_angle, full_turn_angle))

        half_turn_mult = 1.0 - (tf.cast(half_turn, tf.float32) * 0.5)

        full_turn = tf.cast(tf.greater_equal(absolute_angle, half_turn_angle), tf.float32)

        steer = tf.sign(angle_front_to_target) * full_turn * half_turn_mult

        vertical_distance = target_z - bot_position.Z
        should_jump = tf.logical_and(tf.greater(vertical_distance, 100), is_on_ground)

        jump = tf.cast(should_jump, tf.float32)

        ps = tf.logical_and(tf.greater_equal(tf.abs(angle_front_to_target), full_turn_angle),
                            tf.less_equal(distance_to_ball, 2000.0))
        # ps = tf.greater_equal(tf.abs(angle_front_to_target), full_turn_angle)
        power_slide = tf.cast(ps, tf.float32)

        should_not_dodge = tf.cast(tf.greater_equal(distance_to_ball, 500), tf.float32)

        # if jump is 1 then we should not execute a turn
        safe_steer = steer * (1.0 - jump * should_not_dodge)
        return (safe_steer, power_slide, jump)

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
        is_on_ground = tf.cast(values.gamecars[0].bOnGround, tf.bool)
        car_boost = values.gamecars[0].Boost

        bot_yaw = bot_rot.Yaw
        xy_distance = self.distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y)

        # Boost when ball is far enough away
        boost = tf.logical_and(tf.greater_equal(xy_distance, self.distance_from_ball_to_boost / car_boost),
                               tf.greater_equal(car_boost, 10))
        full_throttle = 0.5 * tf.cast(tf.greater(xy_distance, self.distance_from_ball_to_go_fast), tf.float32)
        throttle = full_throttle + tf.constant(0.5)

        blue_goal = tf.constant(-5000.0)
        go_to_ball = tf.cast(tf.less(bot_pos.Y, ball_pos.Y), tf.float32)
        go_to_goal = 1 - go_to_ball
        target_x = ball_pos.X * go_to_ball + self.zero * go_to_goal
        target_y = ball_pos.Y * go_to_ball + blue_goal * go_to_goal
        target_z = ball_pos.Z * go_to_ball + self.zero * go_to_goal

        steer, powerslide, jump = self.aim(bot_pos, bot_yaw,
                                           target_x, target_y, target_z, xy_distance, is_on_ground)

        # Boost on kickoff
        output = [throttle, steer, pitch, yaw, roll, jump, tf.cast(boost, tf.float32), powerslide]
        return output
