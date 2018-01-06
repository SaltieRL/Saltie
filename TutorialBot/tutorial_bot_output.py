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

        pre_change = angle_front_to_target

        # Correct the values
        angle_front_to_target += tf.cond(tf.less(angle_front_to_target, -180.0), lambda: 360.0,
                                         lambda: tf.cond(tf.greater(angle_front_to_target, 180.0), lambda: -360.0,
                                                         lambda: 0.0))

        # angle_front_to_target = tf.Print(angle_front_to_target, [bot_yaw, pre_change, angle_front_to_target], 'angle of bot')

        full_turn_angle = 80
        half_turn_angle = 40
        powerslide_angle = tf.constant(160.0)  # The angle (from the front of the bot to the ball) to start to powerslide.

        st = tf.cond(tf.less(angle_front_to_target, -full_turn_angle), lambda: -1.0,
                     lambda: tf.cond(tf.less(angle_front_to_target, -half_turn_angle), lambda: -0.5,
                     lambda: tf.cond(tf.greater(angle_front_to_target, full_turn_angle), lambda: 1.0,
                     lambda: tf.cond(tf.greater(angle_front_to_target, half_turn_angle), lambda: 0.5,
                                     lambda: 0.0))))
        vertical_distance = target_z - bot_Z
        should_jump = tf.logical_and(tf.greater(vertical_distance, 100), is_on_ground)

        jump = tf.cond(should_jump,
                       lambda: self.true,
                       lambda: self.false)

        ps = tf.cond(tf.greater(tf.abs(angle_front_to_target), powerslide_angle), lambda: self.true,
                     lambda: self.false)
        return (st, ps, jump)

    def get_car_on_ground_direction(self, elements):
        ball_X, ball_Y, ball_Z = elements[0]
        bot_X, bot_Y, bot_Z, bot_yaw = elements[1]
        is_on_ground = elements[2]
        # Blue has their goal at -5000 (Y axis) and orange has their goal at 5000 (Y axis). This means that:
        # - Blue is behind the ball if the ball's Y axis is greater than blue's Y axis
        # - Orange is behind the ball if the ball's Y axis is smaller than orange's Y axis
        st, ps, jump = tf.cond(tf.less(bot_Y, ball_Y),
                         lambda: self.aim(bot_X, bot_Y, bot_Z, bot_yaw, ball_X, ball_Y, ball_Z, is_on_ground),
                         lambda: self.aim(bot_X, bot_Y, bot_Z, bot_yaw, 0.0, -5000.0, 0.0, is_on_ground))

        return (st, ps, jump)

    def hand_kickoff(self, elements):
        throttle, steer, powerslide, jump = elements[0]
        is_kickoff, decomposed_elements = elements[1]

        ball_X, ball_Y, ball_Z = decomposed_elements[0]
        bot_X, bot_Y, bot_Z, bot_yaw = decomposed_elements[1]
        is_on_ground = decomposed_elements[2]

        throttle = tf.cond(is_kickoff,
                           lambda: self.one,
                           lambda: throttle)

        steer, powerslide, jump = tf.cond(is_kickoff,
                                    lambda: self.aim(bot_X, bot_Y, bot_Z, bot_yaw, ball_X, ball_Y, ball_Z, is_on_ground),
                                    lambda: (steer, powerslide, jump))
        return (throttle, steer, powerslide, jump)

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

        throttle = tf.cast(is_on_ground, tf.float32)

        # Boost when ball is far enough away
        boost = tf.logical_and(tf.greater(xy_distance, self.distance_from_ball_to_boost),
                               tf.greater(car_boost, 34))

        elements = [(ball_pos.X, ball_pos.Y, ball_pos.Z),
                    (bot_pos.X, bot_pos.Y, bot_pos.Z, bot_yaw),
                    (is_on_ground)]

        steer, powerslide, jump = tf.map_fn(self.get_car_on_ground_direction, elements,
                                            dtype=(tf.float32, tf.float32, tf.float32))

        # Boost on kickoff

        is_kickoff = tf.logical_and(tf.equal(ball_pos.X, 0.0), tf.equal(ball_pos.Y, 0.0))

        elements = [(throttle, steer, powerslide, jump),
                    (is_kickoff, elements)]


        throttle, steer, powerslide, jump = tf.map_fn(self.hand_kickoff, elements,
                                                      dtype=(tf.float32, tf.float32, tf.float32, tf.float32))

        output = [throttle, steer, pitch, yaw, roll, jump, tf.cast(boost, tf.float32), powerslide]
        return output
