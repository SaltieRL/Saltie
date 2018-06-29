import io
import json
import random
import zipfile

import requests

from model_holder.base_model_holder import BaseModelHolder
from trainer.base_trainer import BaseTrainer


class DownloadTrainer(BaseTrainer):
    def __init__(self, model_holder: BaseModelHolder):
        super().__init__(model_holder)
