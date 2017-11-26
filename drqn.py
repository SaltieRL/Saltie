import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" # required since Rocket League uses GPU
import itertools
import random

import numpy as np
from keras import backend as K
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf

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

        self.state_size = (118,)
        self.number_of_actions = len(self.options)
        self.epsilon = 0.1  # chance of selecting random state, also known as exploration factor
        self.mbsz = 32  # microbatch size
        self.discount = 0.9
        self.memory = 50  # size of memory to feed to network
        self.save_name = 'saltierl'
        self.states = []
        self.actions = []
        self.rewards = []
        self.experience = []
        self.i = 1  # iteration number
        self.save_freq = 100  # how often to save checkpoint, in iterations
        self.total_cost = 0
        self.total_reward = 0
        with tf.device('/cpu:0'):
            self.build_functions()
            self.new_episode()

    def build_model(self):
        S = Input(shape=self.state_size)
        h = Dense(512, activation='relu')(S)
        h = Dense(256, activation='relu')(h)
        h = Dense(256, activation='relu')(h)
        V = Dense(self.number_of_actions)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print("Loading from {}.h5".format(self.save_name))
        except:
            print("Training a new model")

    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        A = Input(shape=(1,), dtype='float32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        self.build_model()
        self.value_fn = K.function([S], [self.model(S)])

        VS = self.model(S)
        VNS = K.stop_gradient(self.model(NS))
        future_value = tf.multiply(tf.to_float(1 - T), tf.reduce_max(VNS, axis=1))
        discounted_future_value = tf.multiply(self.discount, future_value)
        target = tf.add(R, discounted_future_value)
        print ('cost shape', VS.shape, A.shape, target.shape)
        cost = np.mean(((VS[:, A] - target)**2))
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = K.function([S, NS, A, R, T], [cost], updates=updates)

    def new_episode(self):
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def observe(self, reward):
        self.rewards[-1].append(reward)
        return self.iterate()

    def iterate(self):
        N = len(self.states)
        S = np.zeros((self.mbsz,) + self.state_size)
        NS = np.zeros((self.mbsz,) + self.state_size)
        A = np.zeros((self.mbsz, 1), dtype=np.int32)
        R = np.zeros((self.mbsz, 1), dtype=np.float32)
        T = np.zeros((self.mbsz, 1), dtype=np.int32)
        for i in range(self.mbsz):
            episode = random.randint(max(0, N - 50), N - 1)
            num_frames = len(self.states[episode])
            frame = random.randint(0, num_frames - 1)
            S[i] = self.states[episode][frame]
            T[i] = 1 if frame == num_frames - 1 else 0
            if frame < num_frames - 1:
                NS[i] = self.states[episode][frame + 1]
            A[i] = self.actions[episode][frame]
            R[i] = self.rewards[episode][frame]
        cost = self.train_fn([S, NS, A, R, T])
        return cost

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
            print('State size', len(game_inputs))
            with tf.device('/cpu:0'):
                self.states[-1].append(game_inputs)
                values = self.value_fn([np.array(game_inputs).reshape((1, -1))])
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.number_of_actions)
                else:
                    action = np.array(values).argmax()
                self.actions[-1].append(action)
                print(ball_x, ball_y)
                reward = 0.1
                self.total_cost += self.observe(reward)
                self.total_reward += reward
            return self.options[action]

        return random.choice(self.options)
