import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" # required since Rocket League uses GPU
import input_formatter
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
        self.input = input_formatter.InputFormatter(team, index)
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
            #self.build_functions()
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
        """if gameTickPacket.gameInfo.bRoundActive:  # game is currently running
            game_inputs = self.input.create_input_array(game_tick_packet)

            for i in range(gameTickPacket.numBoosts - 1):
                game_inputs.append(gameTickPacket.gameBoosts[i].bActive)
                game_inputs.append(gameTickPacket.gameBoosts[i].Timer)
            print('State size', len(game_inputs))
            with tf.device('/cpu:0'):
          #      self.states[-1].append(game_inputs)
          #      values = self.value_fn([np.array(game_inputs).reshape((1, -1))])
          #      if np.random.random() < self.epsilon:
          #          action = np.random.randint(self.number_of_actions)
          #      else:
          #          action = np.array(values).argmax()
          #      self.actions[-1].append(action)
          #      print(ball_x, ball_y)
          #      reward = 0.1
          #      self.total_cost += self.observe(reward)
          #      self.total_reward += reward
          @  return self.options[action]
          """

        return random.choice(self.options)
