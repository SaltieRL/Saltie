from rlbot.training.training import Grade, Fail
from rlbot.utils.structures.game_data_struct import FieldInfoPacket
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_graders.goal_grader import PassOnGoalForAllyTeam
from rlbottraining.common_graders.rl_graders import FailOnBallOnGroundAfterTimeout
from rlbottraining.common_graders.timeout import FailOnTimeout
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class PartialFail(Fail):
    loss: float

    def __repr__(self):
        return f'{super().__repr__()} Loss = {self.loss}'


class CarBallGoalGrader(FailOnBallOnGroundAfterTimeout):
    loss = math.inf

    def __init__(self, max_duration_seconds: float, ally_team: int = 0):
        super().__init__(max_duration_seconds)
        self.ally_team = ally_team

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        self.loss = min(self.loss, self.get_loss(tick))

        grade = super().on_tick(tick)
        if grade is None:
            return None
        assert isinstance(grade, FailOnTimeout.FailDueToTimeout)
        return PartialFail(self.loss)

    def get_loss(self, tick: TrainingTickPacket) -> float:
        # goal_location = field_info.goals[not self.ally_team].location
        car_location = tick.game_tick_packet.game_cars[0].physics.location
        ball_location = tick.game_tick_packet.game_ball.physics.location
        distance2 = (car_location.x - ball_location.x) ** 2 + \
                    (car_location.y - ball_location.y) ** 2 + \
                    (car_location.z - ball_location.z) ** 2
        return math.sqrt(distance2)


class RocketLeagueCustomStrikerTraining(CompoundGrader):
    def __init__(self, timeout_seconds=4.0, ally_team=0):
        super().__init__([
            PassOnGoalForAllyTeam(ally_team),
            CarBallGoalGrader(timeout_seconds, ally_team)
        ])
