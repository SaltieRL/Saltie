from rlbottraining.exercise_runner import run_playlist
from training.linkuru_playlist import make_default_playlist


if __name__ == '__main__':
    while True:
        result_iter = run_playlist(make_default_playlist())
        results = list(result_iter)
