from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.replay.replay_format import GeneratedHit


class ShotInputFormatter(BaseInputFormatter):
    def get_input_state_dimension(self):
        pass

    def create_input_array(self, input_data: GeneratedHit, batch_size=1):
        result = []
        hit = input_data.get_hit()
        frame = hit.frame_number
        hit_frame = input_data.get_replay().get_pandas().loc[frame]
        result = input_data.get_replay().get_pandas()
        index = input_data.get_replay().get_pandas().loc[frame].name
        return result
