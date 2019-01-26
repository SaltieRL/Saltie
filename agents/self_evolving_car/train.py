import math
import sys
import os
import matplotlib.pyplot as plt
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from examples.levi.input_formatter import LeviInputFormatter
from examples.levi.output_formatter import LeviOutputFormatter
from framework.self_evolving_car.genetic_algorithm import GeneticAlgorithm


class SelfEvolvingCar(BaseAgent):
    """This agent uses neuro-evolution to train the Levi torch model to perform aerials in Rocket League.
        first, the algorithm runs each model with randomly generated parameters. It then calculates each bots' fitness
        by determining it's minimum distance to the ball. Next, it clones the bot with the best fitness to the rest of
        the network, and uses a mutation function to guarantee diversity in the population. Make sure to change match
        length and points to unlimited and disable goal reset and enable instant start"""

    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this is for separate process imports
        from examples.levi.torch_model import SymmetricModel
        self.ga = GeneticAlgorithm()
        self.Model = SymmetricModel
        self.torch = torch
        self.controller_state = SimpleControllerState()
        self.frame = 0  # frame counter for timed reset
        self.brain = 0  # bot counter for generation reset
        self.pop = 10  # population for bot looping
        self.num_best = 5
        self.gen = 0
        self.pos = 0
        self.max_frames = 5000
        self.bot_list = [self.Model() for _ in range(self.pop)]  # list of Individual() objects
        self.bot_list[-self.num_best:] = [self.Model()] * self.num_best  # make sure last bots are the same
        self.bot_fitness = [0] * self.pop
        self.fittest = None  # fittest object
        self.mut_rate = 0.2  # mutation rate
        self.distance_to_ball = [math.inf] * self.max_frames  # set high for easy minimum
        self.input_formatter = LeviInputFormatter(team, index)
        self.output_formatter = LeviOutputFormatter(index)

    def initialize_agent(self):
        self.reset()  # reset at start

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # neural net inputs
        inputs = self.input_formatter.create_input_array([packet])
        inputs = [self.torch.from_numpy(x).float() for x in inputs]

        # fitness function
        my_car = packet.game_cars[self.index]
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z
        self.distance_to_ball[self.frame] = math.sqrt(
            distance_to_ball_x ** 2 + distance_to_ball_y ** 2 + distance_to_ball_z ** 2)

        # render results
        action_display = "GEN: " + str(self.gen + 1) + " | BOT: " + str(self.brain)
        draw_debug(self.renderer, action_display)

        # stop evolving when ball is touched
        # if packet.game_ball.latest_touch.player_name == "Self-Evolving-Car":
        #  self.mut_rate = 0

        # game state
        car_state = CarState(boost_amount=100)
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(0, -1000, 1200),
                                       angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

        # neural net outputs
        with self.torch.no_grad():
            outputs = self.bot_list[self.brain].forward(*inputs)
        self.controller_state = self.output_formatter.format_model_output(outputs, [packet])[0]

        # kill
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950
            or my_car.physics.location.x < -4000 or my_car.physics.location.x > 4000
            or my_car.physics.location.y > 5000 or my_car.physics.location.y < -5000) \
                and self.frame > 50:
            self.frame = self.max_frames

        # loops
        self.frame = self.frame + 1

        if self.frame >= self.max_frames:
            self.frame = 0
            self.bot_fitness[self.brain] = self.ga.calc_fitness(self.distance_to_ball)
            self.distance_to_ball = [math.inf] * self.max_frames
            self.brain += 1  # change bot every reset
            self.controller_state = SimpleControllerState()  # reset controller
            self.reset()  # reset at start

        if self.brain >= self.pop:
            self.gen += 1
            self.brain = 0  # reset bots after all have gone

            self.bot_fitness[self.num_best:] = [self.ga.avg_best_fitness(self.bot_fitness[self.num_best:])
                                                for _ in self.bot_fitness[self.num_best:]]
            self.fittest = self.ga.calc_fittest(self.bot_fitness)

            # print generation info
            self.logger.info("")
            self.logger.info("     GEN = {}".format(self.gen))
            self.logger.info("-------------------------")
            self.logger.info("FITTEST = BOT {}".format(self.fittest))
            self.logger.info("------FITNESS = {}".format(self.bot_fitness[self.fittest]))

            for i in range(len(self.bot_list)):
                self.logger.info("FITNESS OF BOT {} = {}".format(i, self.bot_fitness[i]))

            # NE functions
            self.ga.crossover(self.bot_list[self.fittest], self.bot_list)
            self.ga.mutate(self.bot_list[:self.num_best], self.mut_rate)
            self.reset()  # reset because of delay

        return self.controller_state

    def reset(self):
        """Resets game data after each genome"""
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                       angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0, 0, 0), rotation=Rotator(45, 90, 0),
                                             location=Vector3(0.0, -4608, 500), angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)


def draw_debug(renderer, action_display):
    renderer.begin_rendering()
    renderer.draw_string_2d(10, 10, 4, 4, action_display, color=renderer.white())
    renderer.end_rendering()
