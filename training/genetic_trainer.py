from rlbottraining.exercise_runner import run_playlist
from rlbottraining.training_exercise import TrainingExercise

from rlbot.matchcomms.common_uses.set_attributes_message import make_set_attributes_message
from rlbot.matchcomms.common_uses.reply import send_and_wait_for_replies
from rlbot.training.training import Grade, Pass
from rlbot.utils.logging_utils import get_logger
from rlbot.setup_manager import setup_manager_context

from typing import Optional, Callable

from torch.nn import Module

import io
from multiprocessing.reduction import ForkingPickler
import pickle
import torch
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)  # this is for first process imports

from training.linkuru_playlist import make_default_playlist
from examples.levi.torch_model import SymmetricModel


def create_on_briefing(send_model: Module) -> Callable:
    def on_briefing(self: TrainingExercise) -> Optional[Grade]:
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(send_model)
        _ = send_and_wait_for_replies(self.get_matchcomms(), [
            make_set_attributes_message(0, {'model_hex': buf.getvalue().hex()}),
        ])
        return None
    return on_briefing


if __name__ == '__main__':
    logger = get_logger('genetic algorithm')

    model = SymmetricModel()
    model.load_state_dict(torch.load(f'exercise_0.mdl'))
    model.share_memory()
    playlist = make_default_playlist(create_on_briefing(model))[0:1]

    with setup_manager_context() as setup_manager:
        while True:
            model.load_state_dict(SymmetricModel().state_dict())

            result = next(run_playlist(playlist, setup_manager=setup_manager))
            logger.info(result)

            if isinstance(result.grade, Pass):
                torch.save(model.state_dict(), f'exercise_{result.reproduction_info.playlist_index}.mdl')
                break
