from rlbottraining.exercise_runner import run_playlist
from rlbottraining.training_exercise import TrainingExercise

from training.linkuru_playlist import make_default_playlist

from rlbot.matchcomms.common_uses.set_attributes_message import make_set_attributes_message
from rlbot.matchcomms.common_uses.reply import send_and_wait_for_replies
from rlbot.training.training import Grade

from typing import Optional, Callable

from examples.levi.torch_model import SymmetricModel
from torch.nn import Module

import io
from multiprocessing.reduction import ForkingPickler
import pickle


def create_on_briefing(send_model: Module) -> Callable:
    def on_briefing(self: TrainingExercise) -> Optional[Grade]:
        send_model.share_memory()
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(send_model)
        _ = send_and_wait_for_replies(self.get_matchcomms(), [
            make_set_attributes_message(0, {'model_hex': buf.getvalue().hex()}),
        ])
        return None
    return on_briefing


if __name__ == '__main__':
    model = SymmetricModel()
    playlist = make_default_playlist(create_on_briefing(model))
    while True:
        result_iter = run_playlist(playlist)
        results = list(result_iter)
