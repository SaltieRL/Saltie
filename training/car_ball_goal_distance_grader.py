from rlbot.training.training import Grade, Pass, Fail
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_graders.goal_grader import PassOnGoalForAllyTeam
from rlbottraining.common_graders.rl_graders import FailOnBallOnGroundAfterTimeout
from typing import Optional
from dataclasses import dataclass


@dataclass
class PartialFail(Fail):
    loss: float

    def __repr__(self):
        return f'FAIL: loss = {self.loss}'


class CarBallGoalGrader(FailOnBallOnGroundAfterTimeout):
    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        return super().on_tick(tick)


class RocketLeagueCustomStrikerTraining(CompoundGrader):
    def __init__(self, timeout_seconds=4.0, ally_team=0):
        super().__init__([
            PassOnGoalForAllyTeam(ally_team),
            CarBallGoalGrader(timeout_seconds)
        ])
