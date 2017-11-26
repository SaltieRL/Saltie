

class RewardManager:
    previous_reward = 0
    previous_action = 0
    previous_score = 0
    previous_enemy_goals = 0
    previous_team_goals = 0
    previous_game_score = [0, 0]


    def __init__(self, name, team, index, input_formatter):
        self.name = name
        self.team = team
        self.index = index
        self.input_converter = input_formatter

    def update_from_packet(self, packet):
        self.previous_score = packet.gamecars[self.index].Score
        self.previous_game_score = self.input_converter.score_info
        self.previous_ball_location = packet.gameball.Location
        self.previous_car_location = packet.gamecars[self.index].Location

    def calculate_goal_reward(self):
        """
        :return: change in my team goals - change in enemy team goals should always be 1, 0, -1
        """
        return (self.input_converter.score_info[0] - self.previous_game_score[0]) - \
               (self.input_converter.score_info[0] - self.previous_game_score[0])

    def calculate_score_reward(self, packet):
        """
        :return: change in score.  More score = more reward
        """
        return (packet.gamecars[self.index].Score - self.previous_score) / 100.0

    def get_distance(self, location1, location2):
        return sqrt((location1.X - location2.X)**2 +
                    (location1.Y - location2.Y)**2 +
                    (location1.Z - location2.Z)**2)

    def calculate_ball_follow_change_reward(self, packet):
        """
        When the car moves closer to the ball it gets a reward
        When it moves further it gets punished
        """
        currentDistance = self.get_distance(packet.gamecars[self.index].Location, packet.gameball.Location)
        previousDistance = self.get_distance(self.previous_car_location, self.previous_ball_location)
        #moving faster = bigger reward or bigger punishment
        return previousDistance - currentDistance

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
        reward = 0
        score = packet.gamecars[self.index].Score.Score
        diff = score - self.previous_score
        self.previous_score = score
        reward += diff / 100.0 # max per frame is 100, from a goal, normalizes it
        enemy_index = 1 if self.index == 0 else 1
        if (self.previous_enemy_goals - packet.gamecars[enemy_index].Score.Goals) or (self.previous_owngoals - packet.gamecars[self.index].Score.OwnGoals):
            reward -= 1.0
        return reward
