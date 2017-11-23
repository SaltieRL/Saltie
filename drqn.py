import math
import tensorflow as tf
import itertools
import numpy as np
class Agent:
    def __init__(self, name, team, index):
        self.name = name
        self.team = team  # 0 towards positive goal, 1 towards negative goal.
        self.index = index

        self.timeseries = []
        self.batch_size = 2
        self.time_steps = 5
        self.lstm = tf.contrib.rnn.BasicLSTMCell(256)

        self.hidden_state = self.current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        self.state = self.hidden_state, self.current_state
        self.session = tf.Session()

        # Value	Value Description
        # fThrottle	-1.0 backwards, 1.0 forwards, 0.0 not moving.
        # fSteer	-1.0 left, 1.0 right, 0.0 no turn.
        # fPitch	Like a forward or backward dodge. 0.0 not rotating.
        # fYaw	Like steering but while in midair. 0.0 not rotating.
        # fRoll	Like a side dodge. 0.0 not rotating.
        # bJump	True means jump is held.
        # bBoost	True means boost is held.
        # bHandbrake	True means handbrake is held.
        throttle = np.arange(-1, 1, 0.2)
        steer = np.arange(-1, 1, 0.2)
        pitch = np.arange(-1, 1, 0.2)
        yaw = np.arange(-1, 1, 0.2)
        roll = np.arange(-1, 1, 0.2)
        jump = [True, False]
        boost = [True, False]
        handbrake = [True, False]
        option_list = [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        options = list(itertools.product(*a))
        print (len(options))
        # x = tf.placeholder(tf.float32, [None, 10])
        # W = tf.Variable(tf.zeros([10, num_outputs]))
        # b = tf.Variable(tf.zeros([num_outputs]))

    def get_output_vector(self, game_tick_packet):
        gameTickPacket = game_tick_packet
        enemy_index = 0 if self.index == 1 else 1
        inputs = [
            gameTickPacket.gameball.Location.X,
            gameTickPacket.gameball.Location.Y,
            gameTickPacket.gamecars[self.index].Location.X,
            gameTickPacket.gamecars[self.index].Location.Y,
            float(gameTickPacket.gamecars[self.index].Rotation.Pitch),
            float(gameTickPacket.gamecars[self.index].Rotation.Yaw),
            gameTickPacket.gamecars[enemy_index].Location.X,
            gameTickPacket.gamecars[enemy_index].Location.Y,
            float(gameTickPacket.gamecars[enemy_index].Rotation.Pitch),
            float(gameTickPacket.gamecars[enemy_index].Rotation.Yaw),
        ]
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

        output, self.state = self.lstm(self.timeseries[0], self.state)

        return [1.0, -1.0, 0.0, 0.0, 0.0, 0, 0, 0]

