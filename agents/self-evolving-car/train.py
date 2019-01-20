import math
import random
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from input_formatter import LeviInputFormatter
from output_formatter import LeviOutputFormatter
import sys, os


class PythonExample(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this is for separate process imports
        from model import SymmetricModel
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
        # NEURAL NET INPUTS
        inputs = self.input_formatter.create_input_array([packet])
        inputs = [self.torch.from_numpy(x).float() for x in inputs]

        my_car = packet.game_cars[self.index]
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z
        self.distance_to_ball[self.frame] = math.sqrt(distance_to_ball_x**2 + distance_to_ball_y**2 + distance_to_ball_z**2)

        # RENDER RESULTS
        action_display = "GEN: "+str(self.gen+1)+" | BOT: "+str(self.brain)#+" \nTURN: "+str(self.botList[self.brain].Nodes[1].node(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,4,5,6,7))+" \nBOOST: "+str(self.botList[self.brain].Nodes[2].node(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,8,9,10,11))+" \nTHROTTLE: "+str(self.botList[self.brain].Nodes(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,0,1,2,3))

        draw_debug(self.renderer, my_car, packet.game_ball, action_display)

        # # STOP EVOLVING WHEN THE BALL IS TOUCHED
        # if packet.game_ball.latest_touch.player_name == "Self-Evolving-Car":
        #     self.mut_rate = 0

        # GAME STATE
        car_state = CarState(boost_amount=100)
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(0, -1000, 1200), angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

        # NEURAL NET OUTPUTS
        with self.torch.no_grad():
            outputs = self.bot_list[self.brain].forward(*inputs)
        self.controller_state = self.output_formatter.format_model_output(outputs, [packet])[0]

        # KILL
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or my_car.physics.location.x < -4000 or my_car.physics.location.x > 4000 or my_car.physics.location.y > 5000) and self.frame > 50:
            #self.botList[self.brain].fitness = (1/self.frame+1)*100000
            self.frame = self.max_frames

        # LOOPS
        self.frame = self.frame + 1

        if self.frame >= self.max_frames:
            self.frame = 0
            self.calc_fitness()
            self.brain += 1  # change bot every reset
            self.controller_state = SimpleControllerState()  # reset controller
            self.reset()  # reset at start

        if self.brain >= self.pop:
            self.gen += 1
            self.brain = 0  # reset bots after all have gone

            self.avg_best_fitness()
            self.calc_fittest()

            # PRINT GENERATION INFO
            print("")
            print("     GEN = "+str(self.gen))
            print("-------------------------")
            print("FITTEST = BOT " + str(self.fittest))
            print("------FITNESS = " + str(self.bot_fitness[self.fittest]))
            # print("------WEIGHTS = " + str(self.bot_list[self.fittest]))
            for i in range(len(self.bot_list)):
                print("FITNESS OF BOT " + str(i) + " = " + str(self.bot_fitness[i]))

            # NE Functions

            self.selection()
            self.mutate()
            self.reset()  # reset because of delay

        return self.controller_state

    def calc_fitness(self):
        # CALCULATE MINIMUM DISTANCE TO BALL FOR EACH GENOME
        min_distance_to_ball = min(self.distance_to_ball)
        self.bot_fitness[self.brain] = min_distance_to_ball

        self.distance_to_ball = [math.inf] * self.max_frames

        return min_distance_to_ball

    def avg_best_fitness(self):
        # CALCULATE AVG FITNESS OF 5 FITTEST (IDENTICAL) GENOMES
        self.bot_fitness[-self.num_best:] = [sum(self.bot_fitness[-self.num_best:]) / self.num_best] * self.num_best

    def calc_fittest(self):
        temp = math.inf
        for i in range(len(self.bot_list)):
            if self.bot_fitness[i] < temp:
                temp = self.bot_fitness[i]
                self.fittest = i
        return self.fittest

    def reset(self):
        # RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                       angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0,0,0),rotation=Rotator(45, 90, 0),location=Vector3(0.0, -4608,500),angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)

    def selection(self):
        # COPY FITTEST WEIGHTS TO ALL GENOMES
        state_dict = self.bot_list[self.fittest].state_dict()
        for bot in self.bot_list:
            bot.load_state_dict(state_dict)

    def mutate(self):
        # MUTATE FIRST GENOMES
        for i, bot in enumerate(self.bot_list[:-self.num_best]):
            new_genes = self.Model()
            for param, param_new in zip(bot.parameters(), new_genes.parameters()):
                mask = self.torch.rand(param.data.size()) < self.mut_rate / (i + 1)
                param.data[mask] = param_new.data[mask]


def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()
