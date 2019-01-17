import math
import random
import numpy
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class PythonExample(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.frame = 0  # frame counter for timed reset
        self.brain = -1  # bot counter for generation reset
        self.pop = 10  # population for bot looping
        self.out = [None] * self.pop  # output of nets
        self.gen = 0
        self.botList = [] #list of Individual() objects
        self.fittest = Fittest() #fittest object
        self.pos = 2500 #ball position
        self.mutRate = 0.1 #mutation rate
        self.distance_to_ball = [10000]*10000 #set high for easy minumum
        self.nodeNum = 7 #number of nodes for Node() object array for each individual

        #CREATE BOTS AND NETS
        for i in range(self.pop):
            self.botList.append(Individual())
            self.botList[i].create_node()
            self.botList[i].name = "Bot "+str(i)

        #ASSIGN WEIGHTS
        for i in self.botList:
            print("INIT: "+i.name)
            for p in range(0,len(i.weights)):
                i.weights[p] = random.uniform(-self.mutRate,self.mutRate)
                print("----"+str(i.weights[p]))


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        #INIT LOOPS
        if self.frame == 0:
            self.calcFitness()
            self.reset() #reset at start
            self.brain += 1 #change bot every reset
        if self.brain >= self.pop:
            self.gen += 1

            #PRINT GENERATION INFO
            self.avgBestFitness()
            print("")
            print("     GEN = "+str(self.gen))
            print("-------------------------")
            print("FITTEST = "+str(self.botList[self.calcFittest()].name))
            print("------FITNESS = " + str(self.fittest.fitness))
            print("------WEIGHTS = " + str(self.botList[self.fittest.index].weights))
            for i in range(len(self.botList)):
                print("FITNESS OF BOT "+str(i)+" = "+str(self.botList[i].fitness))

            #NE Functions
            self.selection()
            self.mutate()
            self.brain = 0 #reset bots after all have gone

        self.frame = self.frame + 1


        #NEURAL NET INPUTS
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)
        my_car = packet.game_cars[self.index]
        car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
        car_direction = get_car_facing_vector(my_car)
        car_to_ball = ball_location - car_location
        self.distance_to_ball[self.frame] = math.sqrt(pow(my_car.physics.location.x-packet.game_ball.physics.location.x,2)+pow(my_car.physics.location.y-packet.game_ball.physics.location.y,2)+pow(my_car.physics.location.z-packet.game_ball.physics.location.z,2))
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z


        #RENDER RESULTS
        action_display = "GEN: "+str(self.gen+1)+" | BOT: "+str(self.brain)#+" \nTURN: "+str(self.botList[self.brain].Nodes[1].node(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,4,5,6,7))+" \nBOOST: "+str(self.botList[self.brain].Nodes[2].node(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,8,9,10,11))+" \nTHROTTLE: "+str(self.botList[self.brain].Nodes(self.distance_to_ball_x,self.distance_to_ball_y,self.distance_to_ball_z,0,1,2,3))
        if self.frame > 5000:
            self.frame = 0
        draw_debug(self.renderer, my_car, packet.game_ball, action_display)


        #GAME STATE
        car_state = CarState(boost_amount=100)
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(0, -1000, 1200),angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state,cars={self.index: car_state})
        self.set_game_state(game_state)


        #NEURAL NET OUTPUTS
        hidden1 = self.botList[self.brain].Nodes[0].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 0, 1, 2, 3)
        hidden2 = self.botList[self.brain].Nodes[1].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 4, 5, 6, 7)
        hidden3 = self.botList[self.brain].Nodes[2].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 8, 9, 10, 11)
        hidden4 = self.botList[self.brain].Nodes[3].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 12, 13, 14, 15)

        self.controller_state.pitch = self.botList[self.brain].Nodes[4].node(hidden1, hidden2, hidden3, hidden4, 16, 17, 18, 19)
        self.controller_state.yaw = self.botList[self.brain].Nodes[5].node(hidden1, hidden2, hidden3, hidden4, 20, 21, 22, 23)
        if self.botList[self.brain].Nodes[6].node(hidden1, hidden2, hidden3, hidden4, 24, 25, 26, 27) > 0: self.controller_state.boost = True
        else: self.controller_state.boost = False


        #KILL
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or my_car.physics.location.x < -4000 or my_car.physics.location.x > 4000 or my_car.physics.location.y > 5000) and self.frame > 50:
            #self.botList[self.brain].fitness = (1/self.frame+1)*100000
            self.frame = 5000

        return self.controller_state

    def calcFitness(self):

        #CALCULATE MINIMUM DISTANCE TO BALL FOR EACH GENOME
        min_distance_to_ball = 10000000
        for i in self.distance_to_ball:
            if i < min_distance_to_ball:
                min_distance_to_ball = i
        self.botList[self.brain].fitness = min_distance_to_ball
        for i in range(len(self.distance_to_ball)):
            self.distance_to_ball[i] = 100000

        return min_distance_to_ball

    def avgBestFitness(self):
        # CALCULATE AVG FITNESS OF 5 FITTEST (IDENTICAL) GENOMES
        avg = 0
        for i in range(5, len(self.botList)):
            avg += self.botList[i].fitness
        avg /= 5
        for i in range(5, len(self.botList)):
            self.botList[i].fitness = avg

    def calcFittest(self):
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
        #RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                            angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0,0,0),rotation=Rotator(45, 90, 0),location=Vector3(0.0, -4608,500),angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)

    def selection(self):
        #COPY FITTEST WEIGHTS TO ALL GENOMES
        for i in self.botList:
            for p in range(len(i.weights)):
                i.weights[p] = self.botList[self.fittest.index].weights[p]

    def mutate(self):
        #MUTATE FIRST 5 GENOMES
        for i in range(0,5):
            for p in range(0,10):
                mutWeight = random.randint(0, 27)
                self.botList[i].weights[mutWeight] = random.uniform(-self.mutRate, self.mutRate)

class Vector2:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        return Vector2(self.x + val.x, self.y + val.y)

    def __sub__(self, val):
        return Vector2(self.x - val.x, self.y - val.y)

    def correction_to(self, ideal):
        # The in-game axes are left handed, so use -x
        current_in_radians = math.atan2(self.y, -self.x)
        ideal_in_radians = math.atan2(ideal.y, -ideal.x)

        correction = ideal_in_radians - current_in_radians

        # Make sure we go the 'short way'
        if abs(correction) > math.pi:
            if correction < 0:
                correction += 2 * math.pi
            else:
                correction -= 2 * math.pi

        return correction


def get_car_facing_vector(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)

def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()

class Fittest:
    def __init__(self):
        self.weights = [0] * 20
        self.index = 0
        self.fitness = 0

class Individual:
    def __init__(self):
        self.fitness = 0
        self.name = ""
        self.jump_finished = False
        self.weights = [0] * 28
        self.nodeNum = 7
        self.Nodes = []

    def create_node(self):
        for i in range(self.nodeNum):
            self.Nodes.append(Node(self.weights))

class Node(Individual):
    def __init__(self,weights):
        self.weights = weights

    def node(self,input1, input2, input3,input4, weight1, weight2, weight3,weight4=0):
        out = numpy.tanh((input1 * self.weights[weight1])+(input2 * self.weights[weight2])+(input3 * self.weights[weight3])+(input4 * self.weights[weight4]))
        return out

