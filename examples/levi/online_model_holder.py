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

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter


class OnlineModelHolder:
    def __init__(self, model, input_formatter: BaseInputFormatter, output_formatter: BaseOutputFormatter):
        """
        :param model:
        :param input_formatter:
        :param output_formatter:
        """
        import torch
        self.torch = torch
        self.model = model
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter

    def predict(self, prediction_input):
        """
        Predicts an output given the input
        :param prediction_input: The input, this can be anything as it will go through a BaseInputFormatter
        :return:
        """
        arr = self.input_formatter.create_input_array(prediction_input)
        with self.torch.no_grad():
            output = self.model.predict(arr)
        return self.output_formatter.format_model_output(output)