import time
from rlbot import runner
import os

class Global:

    def __init__(self, pop):
        global Ind
        self.pop = pop

    def Instantiate(self):
        Ind = [None] * self.pop
        for i in range(0,self.pop):
            Ind[i] = Individual()
        return Ind
        
class Individual:
    def Spawn(self):
        runner.main()
        
if __name__ == '__main__':

    playerNames = ["Player 1" , "Player2"]
    bots = Global(2)
    bot = bots.Instantiate()
    for i in range(0,2):
        bot[i].Spawn() #start game
        try:
            os.system('taskkill /IM "RocketLeague.exe" /F')
        except Exception(e):
            print(e)
        time.sleep(10)

