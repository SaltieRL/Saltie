from math import sqrt
import copy

class RewardManager:
    previous_reward = 0
    previous_action = 0
    previous_score = 0
    previous_enemy_goals = 0
    previous_team_goals = 0
    previous_game_score = [0, 0]
    previous_car_location = None
    previous_ball_location = None
    previous_saves = 0


    def __init__(self, name, team, index, input_formatter):
        self.name = name
        self.team = team
        self.index = index
        self.input_converter = input_formatter

    def update_from_packet(self, packet):
        self.previous_saves = packet.gamecars[self.index].Score.Saves
        self.previous_score = packet.gamecars[self.index].Score.Score
        self.previous_game_score = self.input_converter.total_score
        self.previous_ball_location = copy.deepcopy(packet.gameball.Location)
        self.previous_car_location = copy.deepcopy(packet.gamecars[self.index].Location)

    def calculate_save_reward(self, packet):
        """
        :return: change in score.  More score = more reward
        """
        return (packet.gamecars[self.index].Score.Saves - self.previous_saves) * 70 / 100

    def calculate_goal_reward(self):
        """
        :return: change in my team goals - change in enemy team goals should always be 1, 0, -1
        """
        return (self.input_converter.total_score[0] - self.previous_game_score[0]) - \
               (self.input_converter.total_score[1] - self.previous_game_score[1])

    def calculate_score_reward(self, packet):
        """
        :return: change in score.  More score = more reward
        """
        return (packet.gamecars[self.index].Score.Score - self.previous_score) / 100.0

    def get_distance(self, location1, location2):
        return sqrt((location1.X - location2.X)**2 +
                    (location1.Y - location2.Y)**2 +
                    (location1.Z - location2.Z)**2)

    def calculate_ball_follow_change_reward(self, packet):
        """
        When the car moves closer to the ball it gets a reward
        When it moves further it gets punished
        """
        if self.previous_car_location is None or self.previous_ball_location is None:
            return 0
        current_distance = self.get_distance(packet.gamecars[self.index].Location, packet.gameball.Location)
        previous_distance = self.get_distance(self.previous_car_location, self.previous_ball_location)
        #moving faster = bigger reward or bigger punishment
        distance_change = (previous_distance - current_distance) / 100.0
        return min(max(distance_change, 0), .1)

    def calculate_move_fast_reward(self, packet):
        """
        The more the car moves the more reward.
        There is no negative reward only zero
        """
        return self.get_distance(packet.gamecars[self.index].Location, self.previous_car_location)

    def calculate_controller_reward(self, controller1, controller2):
        """
        A handcoded control reward so that the closer it gets to the correct output the better for scalers
        """

    def get_reward(self, packet):
        return max(-1.0, min(1.0,
                self.calculate_ball_follow_change_reward(packet) +
                (self.calculate_goal_reward() + self.calculate_score_reward(packet)) / 2 +
                self.calculate_save_reward(packet)))
