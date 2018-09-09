from rlbot.messages.flat import GameTickPacket

from agents.main_agent.base_model_agent import BaseModelAgent
from swarm_trainer.reward_memory import BaseRewardMemory


class OnlineTrainingAgent(BaseModelAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)

    def predict(self, packet: GameTickPacket):
        result = self.model_holder.predict(packet)
        self.model_holder.train_step(packet, result)
        return result
