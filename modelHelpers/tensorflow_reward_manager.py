from conversions import output_formatter
from conversions.input_formatter import get_state_dim_with_features
from modelHelpers import reward_manager
import tensorflow as tf


class TensorflowRewardManager(reward_manager.RewardManager):
    discount_factor = tf.Variable(initial_value=tf.reshape(tf.constant([0.988, 0.3]), [2, 1]))
    last_state = None
    zero_reward = tf.Variable(initial_value=tf.reshape(tf.constant([0.0, 0.0]), [2, 1]))

    def calculate_save_reward(self, current_score_info, previous_score_info):
        """
        :return: change in score.  More score = more reward
        """
        return (current_score_info.Saves - previous_score_info.Saves) / 2.2

    def calculate_goal_reward(self, fame_score_diff):
        """
        :return: change in my team goals - change in enemy team goals should always be 1, 0, -1
        """
        return fame_score_diff

    def calculate_score_reward(self, score_info, previous_score_info):
        """
        :return: change in score.  More score = more reward
        """
        return (score_info.Score - previous_score_info.Score) / 100.0

    def calculate_ball_follow_change_reward(self, current_info, previous_info):
        """
        When the car moves closer to the ball it gets a reward
        When it moves further it gets punished
        """
        current_distance = self.get_distance_location(current_info.car_location, current_info.ball_location)
        previous_distance = self.get_distance_location(previous_info.car_location, previous_info.ball_location)
        # moving faster = bigger reward or bigger punishment
        distance_change = (previous_distance - current_distance) / 100.0
        return tf.minimum(tf.maximum(distance_change, 0), .3)

    def get_distance_location(self, location1, location2):
        return tf.sqrt(tf.pow(location1.X - location2.X, 2) +
                       tf.pow(location1.Y - location2.Y, 2) +
                       tf.pow(location1.Z - location2.Z, 2))

    def calculate_ball_hit_reward(self, has_last_touched_ball, past_has_last_touched_ball):
        return tf.maximum(0.0, has_last_touched_ball - past_has_last_touched_ball) / 2.0

    def get_state(self, array):
        #array = tf.reshape(array, [tf.shape(array)[0], tf.shape(array)[1]])

        score_info = output_formatter.get_score_info(array, output_formatter.GAME_INFO_OFFSET)
        car_location = output_formatter.create_3D_point(array,
                                                        output_formatter.GAME_INFO_OFFSET + output_formatter.SCORE_INFO_OFFSET)

        ball_location = output_formatter.create_3D_point(array,
                                                         output_formatter.GAME_INFO_OFFSET +
                                                         output_formatter.SCORE_INFO_OFFSET +
                                                         output_formatter.CAR_INFO_OFFSET)
        has_last_touched_ball = array[output_formatter.GAME_INFO_OFFSET +
                                      output_formatter.SCORE_INFO_OFFSET +
                                      output_formatter.CAR_INFO_OFFSET - 1]
        result = output_formatter.create_object()
        result.score_info = score_info
        result.car_location = car_location
        result.ball_location = ball_location
        result.has_last_touched_ball = has_last_touched_ball
        return result

    def calculate_reward(self, previous_state_array, current_state_array):
        current_info = self.get_state(current_state_array)
        previous_info = self.get_state(previous_state_array)

        reward = tf.maximum(tf.constant(-1.0),
                        tf.minimum(tf.constant(1.5),
                            self.calculate_goal_reward(current_info.score_info.FrameScoreDiff) +
                            self.calculate_score_reward(current_info.score_info, previous_info.score_info)) +
             self.calculate_save_reward(current_info.score_info, previous_info.score_info) +
             self.calculate_ball_hit_reward(current_info.has_last_touched_ball, previous_info.has_last_touched_ball)) * 2
        ball_reward = self.calculate_ball_follow_change_reward(current_info, previous_info)
        reward = tf.stack([reward, ball_reward])
        printed_reward = tf.Print(reward, [reward], message='rewards ', first_n=1000)
        return printed_reward

    def create_reward_graph(self, game_input):
        if self.last_state is None:
            self.no_previous_state = tf.Variable(tf.constant(True))
            self.last_state = tf.Variable(tf.zeros([get_state_dim_with_features(), 1]), dtype=tf.float32)

        discounted_reward = tf.Variable(self.zero_reward)

        new_no_previous_state = tf.Variable(tf.constant(False))
        new_last_state = tf.Variable(tf.zeros([get_state_dim_with_features(), 1]), dtype=tf.float32)
        length = tf.Variable(tf.size(game_input))
        discounted_rewards = tf.zeros((tf.shape(game_input)[0], tf.constant(1)), name='discounted_rewards')
        counter = tf.Variable(length)
        tf.while_loop(lambda counter, _, _1, _2, _3, _4, _5, _6 : tf.greater_equal(counter, 0), self.in_loop,
                      (counter, self.no_previous_state, self.last_state,
                       game_input, discounted_rewards, discounted_reward,
                       new_last_state, new_no_previous_state),
                      parallel_iterations=1, back_prop=False)

        # reset the values
        tf.assign(self.no_previous_state, new_no_previous_state)
        tf.assign(self.last_state, new_last_state)
        tf.assign(new_no_previous_state, tf.constant(False))
        tf.assign(new_last_state, tf.Variable(tf.zeros([get_state_dim_with_features(), 1]), dtype=tf.float32))
        discounted_reward.assign(self.zero_reward)
        return discounted_rewards

    def convert_slice_to_state(self, game_input):
        return game_input

    def in_loop(self, counter, has_previous_state, last_state,
                game_input, discounted_rewards, previous_reward,
                new_last_state, new_no_previous_state):

        current_state = self.convert_slice_to_state(tf.slice(game_input, [counter, 0], [1, -1]))

        current_state = tf.reshape(current_state, shape=tf.shape(new_last_state))

        # if counter == len(game_input)
        #     result_new_last_state = current_state
        #     result_new_no_previous_state = True
        result_new_last_state = tf.cond(tf.equal(counter, tf.shape(game_input)[0]), lambda: current_state, lambda: new_last_state)
        result_new_no_previous_state = tf.cond(tf.equal(counter, tf.shape(game_input)[0]), lambda: tf.constant(True), lambda: new_no_previous_state)

        used_last_state = tf.cond(tf.greater(counter, 0),
                                  lambda: self.convert_slice_to_state(tf.slice(game_input, [(counter - 1), 0], [1, -1])),
                                  lambda: last_state)
        used_last_state = tf.reshape(current_state, shape=tf.shape(last_state))

        # if used_last_state is valid
        newest_reward = tf.cond(tf.logical_or(tf.greater(counter, 0), has_previous_state), lambda: self.calculate_reward(used_last_state, current_state),
                                lambda: self.zero_reward)
        new_r = newest_reward + tf.multiply(self.discount_factor, previous_reward)
        index = tf.reshape(counter, [1])
        reward = tf.reshape(tf.reduce_sum(new_r), [1])
        update_tensor = tf.scatter_nd(index, reward, tf.shape(discounted_rewards))
        new_discounted_rewards = discounted_rewards + update_tensor
        new_counter = counter - tf.constant(1)

        return (new_counter, has_previous_state, last_state,
                game_input, new_discounted_rewards, new_r,
                result_new_last_state, result_new_no_previous_state)


