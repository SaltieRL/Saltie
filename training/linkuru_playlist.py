from pathlib import Path
from types import MethodType
import rlbottraining.common_exercises.rl_custom_training_import.rl_importer as rl_importer
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from training.car_ball_goal_distance_grader import CarBallGoalGrader


def make_default_playlist(on_briefing):
    exercises = rl_importer.make_default_playlist()

    for exercise in exercises:
        exercise.match_config.player_configs = [
            PlayerConfig.bot_config(
                Path(__file__).absolute().parent.parent / 'agents' / 'levi_training_agent' / 'levi_training_agent.cfg',
                Team.BLUE
            ),
        ]
        exercise.grader = CarBallGoalGrader(exercise.grader.graders[1].max_duration_seconds)
        exercise.match_config.mutators.boost_amount = "Unlimited"
        exercise.on_briefing = MethodType(on_briefing, exercise)

    return exercises
