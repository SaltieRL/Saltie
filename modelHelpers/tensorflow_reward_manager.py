from conversions.input.input_formatter import get_state_dim_with_features
from modelHelpers import reward_manager
import tensorflow as tf


class TensorflowRewardManager(reward_manager.RewardManager):
    discount_factor = tf.reshape(tf.constant([0.988, 0.3]), [2,])
    last_state = None
    zero_reward = tf.reshape(tf.constant([0.0, 0.0]), [2,])

    def clip_reward(self, reward, lower_bound, upper_bound):
        return tf.minimum(tf.maximum(reward, lower_bound), upper_bound)

    def get_distance_location(self, location1, location2):
        return tf.sqrt(tf.pow(location1.X - location2.X, 2) +
                       tf.pow(location1.Y - location2.Y, 2) +
                       tf.pow(location1.Z - location2.Z, 2))

    def calculate_ball_hit_reward(self, has_last_touched_ball, past_has_last_touched_ball):
        return tf.maximum(0.0, has_last_touched_ball - past_has_last_touched_ball) / 2.0

    def calculate_reward(self, previous_state_array, current_state_array):
        current_info = self.get_state(current_state_array)
        previous_info = self.get_state(previous_state_array)
        rewards = self.calculate_rewards(current_info, previous_info)
        reward = tf.stack([rewards[0], rewards[1]])
        # printed_reward = tf.Print(reward, [reward], message='rewards ', first_n=1000)
        return reward

    def create_reward_graph(self, game_input):
        with tf.name_scope("rewards"):
            resulant_shape = tf.stack([tf.shape(game_input)[0], tf.constant(1)])
            discounted_rewards = tf.fill(resulant_shape, 0.0)
            self.no_previous_state = tf.Variable(tf.constant(False), trainable=False)
            self.last_state = tf.Variable(tf.zeros([get_state_dim_with_features(),]), dtype=tf.float32, trainable=False)

            discounted_reward = self.zero_reward
            length = tf.shape(game_input)[0]
            counter = tf.identity(length, name='counter')

            tf.while_loop(lambda counter, _, _1, _2, _3, _4: tf.greater_equal(counter, 0), self.in_loop,
                          (counter, self.no_previous_state, self.last_state,
                           game_input, discounted_reward, discounted_rewards),
                          parallel_iterations=1, back_prop=False)

            # set the values to be after the first run
            tf.assign(self.no_previous_state, tf.constant(True))
            tf.assign(self.last_state, game_input[length - 1])
            return discounted_rewards

    def convert_slice_to_state(self, game_input):
        return game_input

    def in_loop(self, counter, has_previous_state, last_state,
                game_input, previous_reward, discounted_rewards):

        new_counter = counter - 1

        # if counter == len(game_input)
        #     result_new_last_state = current_state
        #     result_new_no_previous_state = True
        used_last_state = tf.cond(tf.greater_equal(new_counter, 0),
                                  lambda: game_input[new_counter],
                                  lambda: last_state)

        # if used_last_state is valid
        newest_reward = tf.cond(tf.logical_or(tf.greater_equal(new_counter, 0), has_previous_state),
                                lambda: self.calculate_reward(used_last_state, game_input[counter]),
                                lambda: self.zero_reward)
        new_r = newest_reward + tf.multiply(self.discount_factor, previous_reward)

        reward = new_r[0] + new_r[1]
        update_tensor = tf.scatter_nd([counter], [reward], tf.shape(discounted_rewards))

        return (new_counter, has_previous_state, last_state,
                game_input, new_r, update_tensor)
