import itertools
import random

import numpy as np


class Agent:
    def __init__(self, name, team, index):
        self.name = name
        self.team = team  # 0 towards positive goal, 1 towards negative goal.
        self.index = index
        self.enemy_index = 0 if self.index == 1 else 1
        self.timeseries = []
        self.batch_size = 2
        self.time_steps = 5

        # self.hidden_state = self.current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        # self.state = self.hidden_state, self.current_state
        # self.session = tf.Session()

        # Value	Value Description
        # fThrottle	-1.0 backwards, 1.0 forwards, 0.0 not moving.
        # fSteer	-1.0 left, 1.0 right, 0.0 no turn.
        # fPitch	Like a forward or backward dodge. 0.0 not rotating.
        # fYaw	Like steering but while in midair. 0.0 not rotating.
        # fRoll	Like a side dodge. 0.0 not rotating.
        # bJump	True means jump is held.
        # bBoost	True means boost is held.
        # bHandbrake	True means handbrake is held.
        throttle = np.arange(-1, 1, 1)
        steer = np.arange(-1, 1, 1)
        pitch = np.arange(-1, 1, 1)
        yaw = np.arange(-1, 1, 1)
        roll = np.arange(-1, 1, 1)
        jump = [True, False]
        boost = [True, False]
        handbrake = [True, False]
        option_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        self.options = list(itertools.product(*option_list))
        # x = tf.placeholder(tf.float32, [None, 10])
        # W = tf.Variable(tf.zeros([10, num_outputs]))
        # b = tf.Variable(tf.zeros([num_outputs]))

    def get_output_vector(self, game_tick_packet):
        gameTickPacket = game_tick_packet
        if gameTickPacket.gameInfo.bRoundActive:  # game is currently running
            player_x = gameTickPacket.gamecars[self.index].Location.X
            player_y = gameTickPacket.gamecars[self.index].Location.Y
            player_z = gameTickPacket.gamecars[self.index].Location.Z
            player_pitch = float(gameTickPacket.gamecars[self.index].Rotation.Pitch)
            player_yaw = float(gameTickPacket.gamecars[self.index].Rotation.Yaw)
            player_roll = float(gameTickPacket.gamecars[self.index].Rotation.Roll)
            player_speed_x = gameTickPacket.gamecars[self.index].Velocity.X
            player_speed_y = gameTickPacket.gamecars[self.index].Velocity.Y
            player_speed_z = gameTickPacket.gamecars[self.index].Velocity.Z
            player_angular_speed_x = gameTickPacket.gamecars[self.index].AngularVelocity.X
            player_angular_speed_y = gameTickPacket.gamecars[self.index].AngularVelocity.Y
            player_angular_speed_z = gameTickPacket.gamecars[self.index].AngularVelocity.Z
            player_demolished = gameTickPacket.gamecars[self.index].bDemolished
            player_jumped = gameTickPacket.gamecars[self.index].bJumped
            player_double_jumped = gameTickPacket.gamecars[self.index].bDoubleJumped
            player_team = gameTickPacket.gamecars[self.index].Team
            player_boost = gameTickPacket.gamecars[self.index].Boost

            enemy_x = gameTickPacket.gamecars[self.enemy_index].Location.X
            enemy_y = gameTickPacket.gamecars[self.enemy_index].Location.Y
            enemy_z = gameTickPacket.gamecars[self.enemy_index].Location.Z
            enemy_pitch = float(gameTickPacket.gamecars[self.enemy_index].Rotation.Pitch)
            enemy_yaw = float(gameTickPacket.gamecars[self.enemy_index].Rotation.Yaw)
            enemy_roll = float(gameTickPacket.gamecars[self.enemy_index].Rotation.Roll)
            enemy_speed_x = gameTickPacket.gamecars[self.enemy_index].Velocity.X
            enemy_speed_y = gameTickPacket.gamecars[self.enemy_index].Velocity.Y
            enemy_speed_z = gameTickPacket.gamecars[self.enemy_index].Velocity.Z
            enemy_angular_speed_x = gameTickPacket.gamecars[self.enemy_index].AngularVelocity.X
            enemy_angular_speed_y = gameTickPacket.gamecars[self.enemy_index].AngularVelocity.Y
            enemy_angular_speed_z = gameTickPacket.gamecars[self.enemy_index].AngularVelocity.Z
            enemy_demolished = gameTickPacket.gamecars[self.enemy_index].bDemolished
            enemy_jumped = gameTickPacket.gamecars[self.enemy_index].bJumped
            enemy_double_jumped = gameTickPacket.gamecars[self.enemy_index].bDoubleJumped
            enemy_team = gameTickPacket.gamecars[self.enemy_index].Team
            enemy_boost = gameTickPacket.gamecars[self.enemy_index].Boost

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

            # no need for any of these but ball has been hit (kickoff indicator)
            game_timeseconds = gameTickPacket.gameInfo.TimeSeconds
            game_timeremaining = gameTickPacket.gameInfo.GameTimeRemaining
            game_overtime = gameTickPacket.gameInfo.bOverTime
            game_active = gameTickPacket.gameInfo.bRoundActive
            game_ball_hit = gameTickPacket.gameInfo.bBallHasBeenHit
            game_ended = gameTickPacket.gameInfo.bMatchEnded

            game_inputs = [player_x, player_y, player_z, player_pitch, player_yaw, player_roll,
                           player_speed_x, player_speed_y, player_speed_z, player_angular_speed_x,
                           player_angular_speed_y, player_angular_speed_z, player_demolished, player_jumped,
                           player_double_jumped, player_team, player_boost,
                           enemy_x, enemy_y, enemy_z, enemy_pitch, enemy_yaw, enemy_roll,
                           enemy_speed_x, enemy_speed_y, enemy_speed_z, enemy_angular_speed_x,
                           enemy_angular_speed_y, enemy_angular_speed_z, enemy_demolished, enemy_jumped,
                           enemy_double_jumped, enemy_team, enemy_boost,
                           ball_x, ball_y, ball_z, ball_pitch, ball_yaw, ball_roll, ball_speed_x, ball_speed_y,
                           ball_speed_z, ball_angular_speed_x, ball_angular_speed_y, ball_angular_speed_z,
                           ball_acceleration_x, ball_acceleration_y, ball_acceleration_z,
                           game_ball_hit]

            for i in range(gameTickPacket.numBoosts - 1):
                game_inputs.append(gameTickPacket.gameBoosts[i].bActive)
                game_inputs.append(gameTickPacket.gameBoosts[i].Timer)
        # turn -1 is left, 1 is right
        # Value	Value Description
        # fThrottle	-1.0 backwards, 1.0 forwards, 0.0 not moving.
        # fSteer	-1.0 left, 1.0 right, 0.0 no turn.
        # fPitch	Like a forward or backward dodge. 0.0 not rotating.
        # fYaw	Like steering but while in midair. 0.0 not rotating.
        # fRoll	Like a side dodge. 0.0 not rotating.
        # bJump	True means jump is held.
        # bBoost	True means boost is held.
        # bHandbrake	True means handbrake is held.



        return random.choice(self.options)
