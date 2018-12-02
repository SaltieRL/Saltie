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

from rlbot.parsing.custom_config import ConfigHeader, ConfigObject
from agents.swarm.swarm_agent import SwarmAgent
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent, BOT_CONFIG_AGENT_HEADER
from rlbot.utils.class_importer import ExternalClassWrapper
from framework.utils import get_repo_directory


class TeacherAgent(SwarmAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.empty_controller = SimpleControllerState()
        self.teacher = None
        self.teacher_formatter = self.create_output_formatter()
        self.manager_path = None

    def initialize_agent(self):
        super().initialize_agent()
        self.teacher._set_renderer(self.renderer)
        self.teacher.initialize_agent()

    def load_config(self, config_object_header: ConfigHeader):
        super().load_config(config_object_header)
        teacher_path = config_object_header.get('teacher_path')
        self.teacher = ExternalClassWrapper(os.path.join(get_repo_directory(), teacher_path),
                                            BaseAgent).get_loaded_class()(self.name, self.team, self.index)

    def get_output(self, packet):
        """
        Predicts an output given the input
        :param packet: The game_tick_packet
        :return:
        """
        if not packet.game_info.is_round_active:
            return self.empty_controller
        if packet.game_cars[self.index].is_demolished:
            return self.empty_controller

        arr = self.input_formatter.create_input_array([packet], batch_size=1)

        teacher_output = self.teacher.get_output(packet)
        teacher_output, mask = self.teacher_formatter.format_numpy_output(teacher_output, packet)

        assert (arr[0].shape == (1, 3, 9))
        assert (arr[1].shape == (1, 5))
        assert (teacher_output.shape == (1, 13))
        assert (mask.shape == (1, 13))

        data_list = arr.copy()
        data_list.extend([teacher_output, mask])

        self.game_memory.append(data_list)

        output = self.advanced_step(arr, teacher_output)

        # print(teacher_output[0, 5], output[0, 5], mask[0, 5])

        return self.output_formatter.format_model_output(output, [packet], batch_size=1)[0]

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        super(TeacherAgent, TeacherAgent).create_agent_configurations(config)
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('teacher_path', str, default=os.path.join('agents', 'cool_atba', 'cool_atba_agent.py'),
                         description='Path to the teacher bot')

    def advanced_step(self, arr, teacher_output):
        raise NotImplementedError()

    def create_output_formatter(self):
        raise NotImplementedError

    def create_input_formatter(self):
        raise NotImplementedError

    def get_manager_path(self):
        raise NotImplementedError
