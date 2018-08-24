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

from agents.swarm.swarm_agent import SwarmAgent
from examples.levi.online_model_holder import OnlineModelHolder
from examples.levi.output_formatter import LeviOutputFormatter
from examples.levi.input_formatter import LeviInputFormatter


class LeviAgent(SwarmAgent):
    model_holder = None

    def initialize_agent(self):
        self.model_holder = OnlineModelHolder(self.create_model(),
                                              self.create_input_formatter(),
                                              self.create_output_formatter())

    def create_model(self):
        from examples.levi.torch_model import SymmetricModel
        model = SymmetricModel()
        return model

    def create_input_formatter(self):
        return LeviInputFormatter(self.team, self.index)

    def create_output_formatter(self):
        return LeviOutputFormatter(self.index)
