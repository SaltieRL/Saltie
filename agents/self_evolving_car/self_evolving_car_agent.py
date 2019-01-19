import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from examples.self_evolving_car.input_formatter import SelfEvolvingCarInputFormatter
from examples.self_evolving_car.output_formatter import SelfEvolvingCarOutputFormatter
from examples.levi.torch_model import SymmetricModel
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import math
import random
import numpy
import torch


class SelfEvolvingCar(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.torch = torch
        self.controller_state = SimpleControllerState()
        self.frame = 0  # frame counter for timed reset
        self.brain = -1  # bot counter for generation reset
        self.pop = 10  # population for bot looping
        self.out = [None] * self.pop  # output of nets
        self.brain = -1
        self.gen = 0
        self.pos = 0
        self.botList = []  # list of Individual() objects
        self.fittest = Fittest()  # fittest object
        self.mutRate = 0.1  # mutation rate
        self.distance_to_ball = [10000] * 10000  # set high for easy minumum
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()

    def initialize_agent(self):
        # CREATE BOTS AND NETS
        for i in range(self.pop):
            self.botList.append(Individual())
            self.botList[i].name = "Bot " + str(i)

    def create_input_formatter(self):
        return SelfEvolvingCarInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return SelfEvolvingCarOutputFormatter(self.index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        # INIT LOOPS
        if self.frame == 0:
            self.calc_fitness()
            self.reset()  # reset at start
            self.brain += 1  # change bot every reset
        if self.brain >= self.pop:
            self.gen += 1
            self.brain = 0

            # PRINT GENERATION INFO
            self.avg_best_fitness()
            print("")
            print("     GEN = " + str(self.gen))
            print("-------------------------")
            print("FITTEST = " + str(self.botList[self.calc_fittest()].name))
            print("------FITNESS = " + str(self.fittest.fitness))
            for i in range(len(self.botList)):
                print("FITNESS OF BOT " + str(i) + " = " + str(self.botList[i].fitness))
                print(self.botList[i].model.parameters())

            # NE Functions
            self.selection()
            self.mutate()
            self.brain = 0  # reset bots after all have gone

        self.frame = self.frame + 1
        if self.frame > 5000:
            self.frame = 0

        # NEURAL NET INPUTS
        arr = self.input_formatter.create_input_array([packet], batch_size=1)  # formats packet and returns numpy array
        arr = [self.torch.from_numpy(x).float() for x in arr]  # cast numpy input array to tensor

        # NEURAL NET OUTPUTS
        out = self.botList[self.brain].run_model(arr)  # runs indexed model
        self.controller_state = self.output_formatter.format_model_output(out, [packet], batch_size=1)[0]

        # FITNESS
        my_car = packet.game_cars[self.index]
        self.distance_to_ball[self.frame] = math.sqrt(
            pow(my_car.physics.location.x - packet.game_ball.physics.location.x, 2) + pow(
                my_car.physics.location.y - packet.game_ball.physics.location.y, 2) + pow(
                my_car.physics.location.z - packet.game_ball.physics.location.z, 2))

        # RENDER RESULTS
        self.renderer.begin_rendering()
        message = "GEN: " + str(self.gen + 1) + " | BOT: " + str(self.brain)
        self.renderer.draw_string_2d(10, 10, 3, 3, message, self.renderer.white())
        self.renderer.end_rendering()

        # END GENERATION
        if packet.game_ball.latest_touch.player_name == "Self-Evolving-Car":
            self.mutRate = 0

        # GAME STATE
        car_state = CarState(boost_amount=100)
        ball_state = BallState(
            Physics(velocity=Vector3(0, 0, 0), location=Vector3(0, -1000, 1200), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

        # KILL
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or my_car.physics.location.x < -4000
            or my_car.physics.location.x > 4000 or my_car.physics.location.y > 5000) and self.frame > 50:
            self.frame = 5000

        return self.controller_state

    def calc_fitness(self):
        # CALCULATE MINIMUM DISTANCE TO BALL FOR EACH GENOME
        min_distance_to_ball = 10000000
        for i in self.distance_to_ball:
            if i < min_distance_to_ball:
                min_distance_to_ball = i
        self.botList[self.brain].fitness = min_distance_to_ball
        for i in range(len(self.distance_to_ball)):
            self.distance_to_ball[i] = 100000

        return min_distance_to_ball

    def avg_best_fitness(self):
        # CALCULATE AVG FITNESS OF 5 FITTEST (IDENTICAL) GENOMES
        avg = 0
        for i in range(5, len(self.botList)):
            avg += self.botList[i].fitness
        avg /= 5
        for i in range(5, len(self.botList)):
            self.botList[i].fitness = avg

    def calc_fittest(self):
        temp = 1000000
        count = -1
        for i in self.botList:
            count += 1
            if i.fitness < temp:
                temp = i.fitness
                self.fittest.index = count
        self.fittest.fitness = temp
        return self.fittest.index

    def reset(self):
        # RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                       angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0, 0, 0), rotation=Rotator(45, 90, 0),
                                             location=Vector3(0.0, -4608, 500), angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)

    def selection(self):
        # COPY FITTEST WEIGHTS TO ALL GENOMES
        for i in self.botList:
            for param_cur, param_best in zip(i.model.parameters(), self.botList[self.fittest.index].model.parameters()):
                param_cur.data = param_best.data

    def mutate(self):
        # MUTATE FIRST 5 GENOMES
        for i in range(int(len(self.botList)/2)):
            for param in self.botList[i].model.parameters():
                if random.uniform(-1, 1) > 0:
                    scale = 1
                else:
                    scale = -1
                param.data = torch.rand(param.data.size())
                param.data = (param.data/1000) * scale


class Fittest:
    def __init__(self):
        self.index = 0
        self.fitness = 0


class Individual:
    def __init__(self):
        self.fitness = 0
        self.name = ""
        self.jump_finished = False
        self.model = SymmetricModel()

    def run_model(self, inp):
        network_output = self.model.forward(*inp)
        out = network_output.detach().numpy()
        return out
