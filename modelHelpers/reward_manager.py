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

    def calculate_save_reward(self, score_info):
        """
        :return: change in score.  More score = more reward
        """
        return (score_info.Saves - self.previous_saves) / 2.2

    def calculate_goal_reward(self, fame_score_diff):
        """
        :return: change in my team goals - change in enemy team goals should always be 1, 0, -1
        """
        return fame_score_diff

    def calculate_score_reward(self, score_info):
        """
        :return: change in score.  More score = more reward
        """
        return (score_info.Score - self.previous_score) / 100.0

    def calculate_ball_follow_change_reward(self, car_location, ball_location):
        """
        When the car moves closer to the ball it gets a reward
        When it moves further it gets punished
        """
        if self.previous_car_location is None or self.previous_ball_location is None:
            return 0
        current_distance = get_distance_location(car_location, ball_location)
        previous_distance = get_distance_location(self.previous_car_location, self.previous_ball_location)
        # moving faster = bigger reward or bigger punishment
        distance_change = (previous_distance - current_distance) / 100.0
        return min(max(distance_change, -0.05), .3)

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
