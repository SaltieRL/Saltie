from pathlib import Path

import rlbottraining.common_exercises.rl_custom_training_import.rl_importer as rl_importer
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team


def make_default_playlist():
    exercises = (
        rl_importer.make_default_playlist()[10:20]
    )
    for exercise in exercises:
        exercise.match_config.player_configs = [
            PlayerConfig.bot_config(
                Path(__file__).absolute().parent.parent / 'agents' / 'genetic' / 'genetic.cfg',
                Team.BLUE
            ),
        ]

    return exercises
