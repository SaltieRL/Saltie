import tensorflow as tf
import math
from conversions import output_formatter


def get_feature_dim():
    return 5

class TensorflowFeatureCreator:
    unreal_to_degrees = tf.constant(
        1.0 / 65536.0 * 360.0)  # The numbers used to convert unreal rotation units to degrees

    blue_goal_y = tf.constant(-5000.0)
    blue_goal_x = tf.constant(0.0)

    def to_degrees(self, radians):
        return radians * 180 / math.pi

    def generate_features_normalizers(self):
        return [tf.constant([-180.0, 180.0]),
                tf.constant([-180.0, 180.0]),
                tf.constant([-180.0, 180.0]),
                tf.constant([-180.0, 180.0]),
                # max distance (two corners)
                tf.constant([0, 28695])]

    def generate_features(self, input_array):
        advanced_gtp = output_formatter.get_advanced_state(input_array)
        car_info = advanced_gtp.car_info
        ball_info = advanced_gtp.ball_info
        xy_angle_to_ball = self.generate_angle_to_target(car_info.Location.X,
                                                         car_info.Location.Y,
                                                         car_info.Rotation.Yaw,
                                                         ball_info.Location.X,
                                                         ball_info.Location.Y)

        xz_angle_to_ball = self.generate_angle_to_target(car_info.Location.X,
                                                         car_info.Location.Z,
                                                         car_info.Rotation.Pitch,
                                                         ball_info.Location.X,
                                                         ball_info.Location.Z)
        yz_angle_to_ball = self.generate_angle_to_target(car_info.Location.Y,
                                                         car_info.Location.Z,
                                                         car_info.Rotation.Pitch,
                                                         ball_info.Location.Y,
                                                         ball_info.Location.Z)

        xy_angle_to_goal = self.generate_angle_to_target(car_info.Location.X,
                                                         car_info.Location.Y,
                                                         car_info.Rotation.Yaw,
                                                         self.blue_goal_x,
                                                         self.blue_goal_y)

        distance_to_ball = self.get_distance_location(car_info.Location, ball_info.Location)

        return [xy_angle_to_ball, xy_angle_to_goal, xz_angle_to_ball, yz_angle_to_ball, distance_to_ball]

    def get_distance_location(self, location1, location2):
        return tf.sqrt(tf.pow(location1.X - location2.X, 2) +
                       tf.pow(location1.Y - location2.Y, 2) +
                       tf.pow(location1.Z - location2.Z, 2))

    def generate_angle_to_target(self, current_x, current_y, yaw, target_x, target_y):
        bot_yaw = (tf.abs(yaw) % 65536.0) * self.unreal_to_degrees
        # multiple by sign or raw data
        bot_yaw *= tf.sign(yaw)
        y = target_y - current_y
        x = target_x - current_x
        angle_between_bot_and_target = self.to_degrees(tf.atan2(y, x))
        angle_front_to_target = angle_between_bot_and_target - bot_yaw

        angle_front_to_target += (tf.cast(tf.less(angle_front_to_target, -180.0), tf.float32) * 360.0 +
                                  tf.cast(tf.greater(angle_front_to_target, 180.0), tf.float32) * -360.0)

        angle_front_to_target = tf.check_numerics(angle_front_to_target, 'nan angle is being created')

        return angle_front_to_target

    def apply_features(self, model_input):
        transposed_input = tf.transpose(model_input)
        features = self.generate_features(transposed_input)
        features = [tf.expand_dims(feature, axis=1) for feature in features]
        features = [model_input] + features
        new_input = tf.concat(features, axis=1)
        new_input = tf.check_numerics(new_input, 'post features')
        return new_input
