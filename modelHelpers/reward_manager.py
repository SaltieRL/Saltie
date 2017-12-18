from modelHelpers.feature_creator import get_distance_location
from conversions import output_formatter


class RewardManager:
    previous_reward = 0
    previous_score = 0
    previous_enemy_goals = 0
    previous_team_goals = 0
    previous_game_score = [0, 0]
    previous_car_location = None
    previous_ball_location = None
    previous_saves = 0
    has_last_touched_ball = 0

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
        distance_change = (previous_distance - current_distance) / 100.0
        return self.clip_reward(distance_change, 0, .3)

    def get_distance_location(self, location1, location2):
        get_distance_location(location1, location2)

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


    def get_reward(self, array):

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

        # current if you score a goal you will get a reward of 2 that is capped at 1
        # we need some kind of better scaling
        reward = max(-1.0, min(1.0,
                self.calculate_goal_reward(score_info.FrameScoreDiff) +
                self.calculate_ball_follow_change_reward(car_location, ball_location) +
                self.calculate_score_reward(score_info)) +
                self.calculate_save_reward(score_info) +
                self.calculate_ball_hit_reward(has_last_touched_ball, self.has_last_touched_ball)) * 2

        self.previous_saves = score_info.Saves
        self.previous_score = score_info.Score
        self.previous_ball_location = ball_location
        self.previous_car_location = car_location
        self.previous_reward = reward
        self.has_last_touched_ball = has_last_touched_ball

        return reward
