from framework.output_formatter.base_output_formatter import BaseOutputFormatter
from framework.replay.replay_format import GeneratedHit


class ShotOutputFormatter(BaseOutputFormatter):

    def get_model_output_dimension(self):
        return [1]

    def create_array_for_training(self, input_hit: GeneratedHit, batch_size=1):
        return [int(input_hit.get_hit().goal)]
