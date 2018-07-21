from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.logging_utils import get_logger
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.quick_chats import QuickChats

from examples.legacy.legacy_game_input_formatter import LegacyGameInputFormatter
from examples.lstm.lstm_input_formatter import LSTMInputFormatter
from examples.legacy.legacy_output_formatter import LegacyOutputFormatter
from examples.lstm.lstm_output_formatter import LSTMOutputFormatter


class BaseModelAgent(BaseAgent):

    model_holder = None
    controller_state = None
    logger = None

    def initialize_agent(self):
        self.logger = get_logger(self.name)
        # This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

        from examples.example_model_holder import ExampleModelHolder

        self.model_holder = ExampleModelHolder(self.create_model(),
                                               self.create_input_formatter(),
                                               self.create_output_formatter())

        self.model_holder.initialize_model(load=True)
        self.logger.info("Model has been initialized")

    def create_model(self):
        # Models need to be imported locally dues to creation of tensorflow and keras on imports
        from examples.lstm.example_lstm_model import ExampleLSTMModel
        #return ExampleLSTMModel(prediction_mode=True)

        from examples.autoencoder.autoencoder_model import AutoencoderModel
        return AutoencoderModel(compressed_dim=50)

    def create_input_formatter(self):
        return LegacyGameInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LegacyOutputFormatter()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        result = self.model_holder.predict(packet)
        self.controller_state.throttle = max(1, min(-1, result[0]))
        self.controller_state.steer = max(1, min(-1, result[1]))
        self.controller_state.pitch = max(1, min(-1, result[2]))
        self.controller_state.yaw = max(1, min(-1, result[3]))
        self.controller_state.roll = max(1, min(-1, result[4]))
        self.controller_state.jump = max(1, min(0, result[5]))
        self.controller_state.boost = max(1, min(0, result[6]))
        self.controller_state.handbrake = max(1, min(0, result[7]))

        self.logger.info("%s", str(result))

        return self.controller_state
