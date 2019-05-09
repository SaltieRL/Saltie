# MIT License
#
# Copyright (c) 2018 LHolten@Github Hytak#5125@Discord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent
from rlbot.matchcomms.common_uses.set_attributes_message import handle_set_attributes_message
from rlbot.matchcomms.common_uses.reply import reply_to
from queue import Empty
import pickle

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path)  # this is for first process imports

from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter


class LeviAgent(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        sys.path.insert(0, path)  # this is for separate process imports
        import torch
        self.torch = torch
        self.empty_controller = SimpleControllerState()
        self.model = None
        self.model_hex = None
        self.input_formatter = None
        self.output_formatter = None

    def initialize_agent(self):
        self.input_formatter = self.create_input_formatter()
        self.output_formatter = self.create_output_formatter()

    def get_output(self, packet):
        self.handle_messages()

        if not packet.game_info.is_round_active:
            return self.empty_controller
        if packet.game_cars[self.index].is_demolished:
            return self.empty_controller
        if not self.model:
            return self.empty_controller

        arr = self.input_formatter.create_input_array([packet])

        with self.torch.no_grad():
            tensors = [self.torch.from_numpy(x).float() for x in arr]
            assert (tensors[0].size() == (1, 3, 9))
            assert (tensors[1].size() == (1, 5))
            out_tensors = self.model.forward(*tensors)
            new_output, _ = (x.numpy() for x in out_tensors)
            # new_output, _, _, _ = (x.numpy() for x in out_tensors)

        mask = self.output_formatter.get_mask(packet)
        assert (mask.shape == (1, 13))

        controls = self.output_formatter.format_controller_output(new_output[0] * mask[0], packet)

        # game_info_state = GameInfoState(game_speed=3.0)
        # game_state = GameState(game_info=game_info_state)
        # self.set_game_state(game_state)

        return controls

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)

    def handle_messages(self):
        for i in range(100):  # process at most 100 messages per tick.
            try:
                msg = self.matchcomms.incoming_broadcast.get_nowait()
            except Empty:
                break

            if handle_set_attributes_message(msg, self, allowed_keys=['model_hex']):
                reply_to(self.matchcomms, msg)  # Let the sender know we've set the attribute.
                self.model = pickle.loads(bytes.fromhex(self.model_hex))
            else:
                # Ignore messages that are not for us.
                self.logger.debug(f'Unhandled message: {msg}')
