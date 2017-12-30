import math
import tensorflow as tf


class TutorialBotOutput:
    # Constants
    distance_from_ball_to_boost = tf.constant(1500.0)  # Minimum distance to ball for using boost
    powerslide_angle = tf.constant(170.0)  # The angle (from the front of the bot to the ball) to start to powerslide.
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

    def aim(self, bot_X, bot_Y, bot_yaw, target_x, target_y):
        angle_between_bot_and_target = self.to_degrees(
            tf.atan2(tf.subtract(target_y, bot_Y), tf.subtract(target_x, bot_X)))
        angle_front_to_target = angle_between_bot_and_target - bot_yaw

        # Correct the values
        angle_front_to_target += tf.cond(tf.less(angle_front_to_target, -180.0), lambda: 360.0,
                                         lambda: tf.cond(tf.greater(angle_front_to_target, 180.0), lambda: -360.0,
                                                         lambda: 0.0))

        st = tf.cond(tf.less(angle_front_to_target, -10), lambda: -1.0,
                     lambda: tf.cond(tf.greater(angle_front_to_target, 10), lambda: 1.0, lambda: 0.0))

        ps = tf.cond(tf.less(tf.abs(self.to_degrees(angle_front_to_target)), self.powerslide_angle), lambda: self.true,
                     lambda: self.false)
        return (st, ps)

    def get_car_on_ground_direction(self, elements):
        ball_X, ball_Y = elements[0]
        bot_X, bot_Y, bot_yaw = elements[1]
        st, ps = tf.cond(tf.less(bot_Y, ball_Y),
                         lambda: self.aim(bot_X, bot_Y, bot_yaw, ball_X, ball_Y),
                         lambda: self.aim(bot_X, bot_Y, bot_yaw, 0.0, -5000.0))

        return [(st, ps), elements[1]]

    def hand_kickoff(self, elements):
        throttle, steer, powerslide = elements[0]
        is_kickoff, decomposed_elements = elements[1]

        ball_X, ball_Y = decomposed_elements[0]
        bot_X, bot_Y, bot_yaw = decomposed_elements[1]

        throttle = tf.cond(is_kickoff,
                           lambda: self.one,
                           lambda: throttle)

        steer, powerslide = tf.cond(is_kickoff,
                                    lambda: self.aim(bot_X, bot_Y, bot_yaw, ball_X, ball_Y),
                                    lambda: (steer, powerslide))
        return [(throttle, steer, powerslide), (is_kickoff, decomposed_elements)]

    def calculate_loss(self, elements):
        throttle = elements[0]
        is_on_ground = elements[1]
        given_output = elements[2]
        created_output = elements[3]
        steer, powerslide, pitch, jump, boost = created_output

        def output_on_ground():
            # Throttle
            output = tf.losses.absolute_difference(throttle, given_output[0])

            # Steer
            # output += tf.cond(tf.less_equal(tf.abs(steer - given_output[1]), 0.5), lambda: 1, lambda: -1)
            output += tf.losses.absolute_difference(steer, given_output[1])

            # Powerslide
            # output += tf.cond(tf.equal(tf.cast(powerslide, tf.float32), given_output[7]), lambda: 1, lambda: -1)
            output += tf.losses.mean_squared_error(powerslide, given_output[1])
            return output

        def output_off_ground():
            # Pitch
            output = tf.losses.absolute_difference(pitch, given_output[2])
            # output = tf.cond(tf.less_equal(tf.abs(pitch - given_output[2]), 0.5), lambda: 1, lambda: -1)
            return output

        output = tf.cond(is_on_ground, output_on_ground, output_off_ground)

        # Jump
        # output += tf.cond(tf.equal(tf.cast(jump, tf.float32), given_output[5]), lambda: 1, lambda: -1)
        output += tf.losses.mean_squared_error(jump, given_output[5])

        # Boost
        # output += tf.cond(tf.equal(tf.cast(boost, tf.float32), given_output[6]), lambda: 1, lambda: -1)
        output += tf.losses.mean_squared_error(boost, given_output[6])

        return [output, elements[1], elements[2], elements[3]]

    def get_output_vector(self, values, given_output):
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

        # Get car's yaw and convert from Unreal Rotator units to degrees
        bot_yaw = (tf.abs(bot_rot.Yaw) % 65536.0) * self.unreal_to_degrees
        # multiple by sign or raw data
        bot_yaw *= tf.sign(bot_rot.Yaw)

        boost_cond = tf.greater(self.distance(bot_pos.X, bot_pos.Y, ball_pos.X, ball_pos.Y),
                                self.distance_from_ball_to_boost)

        # Boost when ball is far enough away
        boost = boost_cond

        # Blue has their goal at -5000 (Y axis) and orange has their goal at 5000 (Y axis). This means that:
        # - Blue is behind the ball if the ball's Y axis is greater than blue's Y axis
        # - Orange is behind the ball if the ball's Y axis is smaller than orange's Y axis

        elements = [(ball_pos.X, ball_pos.Y),
                    (bot_pos.X, bot_pos.Y, bot_yaw)]

        steer, powerslide = tf.map_fn(self.get_car_on_ground_direction, elements)[0]

        # Boost on kickoff

        is_kickoff = tf.logical_and(tf.equal(ball_pos.X, 0.0), tf.equal(ball_pos.Y, 0.0))

        elements = [(throttle, steer, powerslide),
                    (is_kickoff, elements)]

        boost = tf.logical_or(is_kickoff, tf.cast(boost, tf.bool))

        throttle, steer, powerslide = tf.map_fn(self.hand_kickoff, elements)[0]

        elements = [throttle, values.gamecars[0].bOnGround, given_output, (steer, powerslide, pitch, jump, boost)]

        output = [throttle, steer, pitch, yaw, roll, jump, boost, powerslide]
        loss = tf.map_fn(self.calculate_loss, elements)[0]
        return (loss, output)
