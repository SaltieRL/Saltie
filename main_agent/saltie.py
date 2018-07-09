from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class Saltie(BaseAgent):

    model_holder = None

    def initialize_agent(self):
        #This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

        from examples.legacy.legacy_game_input_formatter import LegacyGameInputFormatter
        from examples.example_keras_model import LegacyKerasModel
        from examples.example_model_holder import ExampleModelHolder
        from examples.legacy.legacy_output_formatter import LegacyOutputFormatter

        self.model_holder = ExampleModelHolder(LegacyKerasModel(),
                                               LegacyGameInputFormatter(self.team, self.index),
                                               LegacyOutputFormatter())

        self.model_holder.initialize_model(load=True)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        result = self.model_holder.predict(packet)
        self.controller_state.throttle = result[0]
        self.controller_state.steer = result[1]
        self.controller_state.pitch = result[2]
        self.controller_state.yaw = result[3]
        self.controller_state.roll = result[4]
        self.controller_state.jump = result[5]
        self.controller_state.boost = result[6]
        self.controller_state.boost = result[7]

        return self.controller_state
