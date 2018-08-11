import os


def get_repo_directory():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
