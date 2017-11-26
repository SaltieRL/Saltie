class InputFormatter:

    def __init__(self, team, index):
        self.team = team
        self.index = index

    def create_input_array(self, gameTickPacket):
        teamMembers = []
        enemies = []
        player_car = self.returnEmtpyPlayerArray()
        for index in range(len(gameTickPacket.gamecars)):
            if index == self.index:
                player_car = self.get_car_info(gameTickPacket, index)
            if gameTickPacket.gamecars[index].Team == self.team:
                teamMembers.append(self.get_car_info(gameTickPacket, index))
            else:
                enemies.append(self.get_car_info(gameTickPacket, index))
        while len(teamMembers) < 2:
            teamMembers.append(self.returnEmtpyPlayerArray())
        while len(enemies) < 3:
            enemies.append(self.returnEmtpyPlayerArray()) 

        ball_data = self.get_ball_info(gameTickPacket)
        game_info = self.get_game_info(gameTickPacket)
        boost_info = self.get_boost_info(gameTickPacket)

        return player_car + ball_data + self.flattenArrays(teamMembers) + self.flattenArrays(enemies) + game_info + boost_info

    def returnEmtpyPlayerArray(self):
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    def get_car_info(self, gameTickPacket, index):
        player_x = gameTickPacket.gamecars[index].Location.X
        player_y = gameTickPacket.gamecars[index].Location.Y
        player_z = gameTickPacket.gamecars[index].Location.Z
        player_pitch = float(gameTickPacket.gamecars[index].Rotation.Pitch)
        player_yaw = float(gameTickPacket.gamecars[index].Rotation.Yaw)
        player_roll = float(gameTickPacket.gamecars[index].Rotation.Roll)
        player_speed_x = gameTickPacket.gamecars[index].Velocity.X
        player_speed_y = gameTickPacket.gamecars[index].Velocity.Y
        player_speed_z = gameTickPacket.gamecars[index].Velocity.Z
        player_angular_speed_x = gameTickPacket.gamecars[index].AngularVelocity.X
        player_angular_speed_y = gameTickPacket.gamecars[index].AngularVelocity.Y
        player_angular_speed_z = gameTickPacket.gamecars[index].AngularVelocity.Z
        player_demolished = gameTickPacket.gamecars[index].bDemolished
        player_jumped = gameTickPacket.gamecars[index].bJumped
        player_double_jumped = gameTickPacket.gamecars[index].bDoubleJumped
        player_team = gameTickPacket.gamecars[index].Team
        player_boost = gameTickPacket.gamecars[index].Boost
        return [player_x, player_y, player_z, player_pitch, player_yaw, player_roll,
                     player_speed_x, player_speed_y, player_speed_z, player_angular_speed_x,
                     player_angular_speed_y, player_angular_speed_z, player_demolished, player_jumped,
                     player_double_jumped, player_team, player_boost]

    def get_game_info(self, gameTickPacket):
        game_ball_hit = gameTickPacket.gameInfo.bBallHasBeenHit
        
        # no need for any of these but ball has been hit (kickoff indicator)
        # game_timeseconds = gameTickPacket.gameInfo.TimeSeconds
        # game_timeremaining = gameTickPacket.gameInfo.GameTimeRemaining
        # game_overtime = gameTickPacket.gameInfo.bOverTime
        # game_active = gameTickPacket.gameInfo.bRoundActive
        # game_ended = gameTickPacket.gameInfo.bMatchEnded
        return [game_ball_hit]

    def get_ball_info(self, gameTickPacket):
        ball_x = gameTickPacket.gameball.Location.X
        ball_y = gameTickPacket.gameball.Location.Y
        ball_z = gameTickPacket.gameball.Location.Z
        ball_pitch = float(gameTickPacket.gameball.Rotation.Pitch)
        ball_yaw = float(gameTickPacket.gameball.Rotation.Yaw)
        ball_roll = float(gameTickPacket.gameball.Rotation.Roll)
        ball_speed_x = gameTickPacket.gameball.Velocity.X
        ball_speed_y = gameTickPacket.gameball.Velocity.Y
        ball_speed_z = gameTickPacket.gameball.Velocity.Z
        ball_angular_speed_x = gameTickPacket.gameball.AngularVelocity.X
        ball_angular_speed_y = gameTickPacket.gameball.AngularVelocity.Y
        ball_angular_speed_z = gameTickPacket.gameball.AngularVelocity.Z
        ball_acceleration_x = gameTickPacket.gameball.Acceleration.X
        ball_acceleration_y = gameTickPacket.gameball.Acceleration.Y
        ball_acceleration_z = gameTickPacket.gameball.Acceleration.Z
        return [ball_x, ball_y, ball_z, ball_pitch, ball_yaw, ball_roll, ball_speed_x, ball_speed_y,
                    ball_speed_z, ball_angular_speed_x, ball_angular_speed_y, ball_angular_speed_z,
                    ball_acceleration_x, ball_acceleration_y, ball_acceleration_z]

    def get_boost_info(self, gameTickPacket):
        game_inputs = []
        for i in range(gameTickPacket.numBoosts - 1):
            game_inputs.append(gameTickPacket.gameBoosts[i].bActive)
            game_inputs.append(gameTickPacket.gameBoosts[i].Timer)
        return game_inputs

    def flattenArrays(self, arrayOfArray):
        return [item for sublist in arrayOfArray for item in sublist]
        
