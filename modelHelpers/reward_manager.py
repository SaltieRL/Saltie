from modelHelpers.feature_creator import get_distance_location
from conversions import output_formatter


class RewardManager:
    previous_info = None

    def calculate_save_reward(self, current_score_info, previous_score_info):
        """
        :return: gets reward for saving! :) more saving more reward
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

    def clip_reward(self, reward, lower_bound, upper_bound):
        return min(upper_bound, max(lower_bound, reward))

    def calculate_ball_follow_change_reward(self, current_info, previous_info):
        """
        When the car moves closer to the ball it gets a reward
        When it moves further it gets punished
        """
        current_distance = self.get_distance_location(current_info.car_location, current_info.ball_location)
        previous_distance = self.get_distance_location(previous_info.car_location, previous_info.ball_location)
        # moving faster = bigger reward or bigger punishment
        distance_change = (previous_distance - current_distance) / 500.0
        return self.clip_reward(distance_change, -0.001, .2)

    def calculate_ball_closeness_reward(self, current_info, previous_info):
        """
        When the car is within a certain distance of the ball it gets a reward for being that close
        """
        current_distance = self.get_distance_location(current_info.car_location, current_info.ball_location)

        # += does not work on tensorflow objects
        # prevents a distance of 0
        current_distance = current_distance + 0.000001
        divided_value = 100.0 / current_distance
        divided_value = divided_value * divided_value
        clipped_value = self.clip_reward(divided_value, 0, .2)
        return clipped_value


    def get_distance_location(self, location1, location2):
        return get_distance_location(location1, location2)

    def calculate_move_fast_reward(self, packet):
        """
        The more the car moves the more reward.
        There is no negative reward only zero
        """
        return get_distance_location(packet.gamecars[self.index].Location, self.previous_car_location)

    def calculate_controller_reward(self, controller1, controller2):
        """
        A handcoded control reward so that the closer it gets to the correct output the better for scalers
        """

    def calculate_ball_hit_reward(self, has_last_touched_ball, past_has_last_touched_ball):
        return max(0, has_last_touched_ball - past_has_last_touched_ball) / 2.0

    def get_state(self, array):
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

    def calculate_rewards(self, current_info, previous_info):
        reward = self.clip_reward(((
                self.calculate_goal_reward(current_info.score_info.FrameScoreDiff) +
                self.calculate_score_reward(current_info.score_info, previous_info.score_info)) +
                self.calculate_save_reward(current_info.score_info, previous_info.score_info) +
                self.calculate_ball_hit_reward(current_info.has_last_touched_ball,
                    previous_info.has_last_touched_ball)),
            -1, 1)
        ball_reward = self.calculate_ball_follow_change_reward(current_info, previous_info) +\
                      self.calculate_ball_closeness_reward(current_info, previous_info)
        return [reward, ball_reward]

    def get_reward(self, array):
        current_info = self.get_state(array)
        rewards = [0.0, 0.0]
        if self.previous_info is not None:
            rewards = self.calculate_rewards(current_info, self.previous_info)
        self.previous_info = current_info
        return rewards

