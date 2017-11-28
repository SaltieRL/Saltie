class OutputFormatter:
    """
    This is a class that takes in an arrray and will return a gametick packet of that value
    """

    def create_output_array(self, array):
        gameTickPacket = self.create_object()
        gameTickPacket.gamecars = []
        total_offset = 0
        game_info, offset = self.get_game_info(array, 0)
        total_offset += offset
        gameTickPacket.gameInfo = game_info
        score_info, offset = self.get_score_info(array, total_offset)
        total_offset += offset
        player_car, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        player_car.Score = score_info
        # always put player car at 0
        gameTickPacket.gamecars.append(player_car)

        ball_info, offset = self.get_ball_info(array, total_offset)
        total_offset += offset
        gameTickPacket.gameball = ball_info

        team_member1, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        if team_member1 is not None:
            gameTickPacket.gamecars.append(team_member1)

        team_member2, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        if team_member2 is not None:
            gameTickPacket.gamecars.append(team_member2)

        enemy1, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        if enemy1 is not None:
            gameTickPacket.gamecars.append(enemy1)

        enemy2, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        if enemy2 is not None:
            gameTickPacket.gamecars.append(enemy2)

        enemy3, offset = self.get_car_info(array, total_offset)
        total_offset += offset
        if enemy3 is not None:
            gameTickPacket.gamecars.append(enemy3)

        gameTickPacket.gameBoosts = self.get_boost_info(array, total_offset)

        return gameTickPacket

    def is_empty_player_array(self, array, index, offset):
        sublist = array[index:index + offset]
        return all(p == 0.0 for p in sublist)

    def create_object(self):
        return lambda: None

    def create_3D_point(self, x, y, z):
        point = self.create_object()
        point.X = x
        point.Y = y
        point.Z = z
        return point

    def create_3D_rotation(self, x, y, z):
        point = self.create_object()
        point.Pitch = x
        point.Yaw = y
        point.Roll = z
        return point

    def get_car_info(self, array, index):
        if self.is_empty_player_array(array, index, 17):
            return None, 17
        car_info = self.create_object()
        car_info.Location = self.create_3D_point(array[index], array[index + 1], array[index + 2])
        car_info.Rotation = self.create_3D_rotation(array[index + 3], array[index + 4], array[index + 5])
        car_info.Velocity = self.create_3D_point(array[index + 6], array[index + 7], array[index + 8])
        car_info.AngularVelocity = self.create_3D_point(array[index + 9], array[index + 10], array[index + 11])
        car_info.bDemolished = (array[12] == 1)
        car_info.bJumped = (array[13] == 1)
        car_info.bDoubleJumped = (array[14] == 1)
        car_info.Team = int(array[15])
        car_info.Boost = array[16]
        return car_info, 17

    def get_game_info(self, array, index):
        game_info = self.create_object()
        game_info.bBallHasBeenHit = (array[index] == 1)

        # no need for any of these but ball has been hit (kickoff indicator)
        # game_timeseconds = gameTickPacket.gameInfo.TimeSeconds
        # game_timeremaining = gameTickPacket.gameInfo.GameTimeRemaining
        # game_overtime = gameTickPacket.gameInfo.bOverTime
        # game_active = gameTickPacket.gameInfo.bRoundActive
        # game_ended = gameTickPacket.gameInfo.bMatchEnded
        return game_info, 1

    def get_ball_info(self, array, index):
        ball_info = self.create_object()
        ball_info.Location = self.create_3D_point(array[index], array[index + 1], array[index + 2])
        ball_info.Rotation = self.create_3D_rotation(array[index + 3], array[index + 4], array[index + 5])
        ball_info.Velocity = self.create_3D_point(array[index + 6], array[index + 7], array[index + 8])
        ball_info.AngularVelocity = self.create_3D_point(array[index + 9], array[index + 10], array[index + 11])
        ball_info.Acceleration = self.create_3D_point(array[index + 12], array[index + 13], array[index + 14])
        return ball_info, 15

    def get_boost_info(self, array, index):
        boost_objects = []
        for i in range(index, len(array), 2):
            boost_info = self.create_object()
            boost_info.bActive = (array[i] == 1)
            boost_info.Timer = array[i + 1]
            boost_objects.append(boost_info)
        return boost_objects

    def get_score_info(self, array, index):
        score_info = self.create_object()
        score_info.Score = array[index]
        score_info.Goals = array[index + 1]
        score_info.OwnGoals = array[index + 2]
        score_info.Assists = array[index + 3]
        score_info.Saves = array[index + 4]
        score_info.Shots = array[index + 5]
        score_info.Demolitions = array[index + 6]
        return score_info, 7

