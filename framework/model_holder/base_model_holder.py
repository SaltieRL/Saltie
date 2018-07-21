from rlbot.utils.logging_utils import get_logger

from framework.input_formatter.base_input_formatter import BaseInputFormatter
from framework.output_formatter.base_output_formatter import BaseOutputFormatter
from framework.model.base_model import BaseModel
from framework.reward_manager.base_reward_manager import BaseRewardManager
from framework.utils import get_repo_directory


class BaseModelHolder:

    use_custom_fit = False
    use_custom_sample_action = False
    model_output = None
    logger = None

    def __init__(self, model: BaseModel, input_formatter: BaseInputFormatter, output_formatter: BaseOutputFormatter,
                 reward_manager: BaseRewardManager=None):
        """

        :param model:
        :param input_formatter:
        :param output_formatter:
        :param reward_manager:
        """
        self.logger = get_logger(str(type(self).__name__))
        self.model = model
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
        self.reward_manager = reward_manager

        self.use_custom_fit = not hasattr(self.model.fit, 'is_native')
        self.use_custom_sample_action = not hasattr(self.model.predict, 'is_native')

    def initialize_model(self, load=False):
        input_layer = self.model.create_input_layer(self.input_formatter)
        hidden_layer = self.model.create_hidden_layers(input_layer=input_layer)
        self.model_output = self.model.create_output_layer(self.output_formatter, hidden_layer=hidden_layer)
        if load:
            self.__load_model_safely()
        self.model.finalize_model()

    def train_step(self, input_array, output_array, rewards=None, batch_size=1):
        """
        Performs a single train step on the data given.
        All data (input, output, rewards) should end up producing arrays of the same length
        :param input_array: Fed as input to the model this is the data that is expected to produce results.
        :param output_array: The expected result that the model should produce.
        :param rewards: Optional, rewards are weighted values to say how strongly a certain action should be copied.
        :param batch_size: How many are in the array
        :return:
        """
        formatted_input = self.input_formatter.create_input_array(input_array, batch_size=batch_size)
        formatted_output = self.output_formatter.create_array_for_training(output_array, batch_size=batch_size)
        if self.reward_manager is not None:
            reward_input = input_array if self.reward_manager.has_input_formatter() else formatted_input
            reward_output = output_array if self.reward_manager.has_output_formatter() else formatted_output
            created_rewards = self.reward_manager.create_reward(reward_input, reward_output,
                                                                existing_rewards=rewards, batch_size=batch_size)
        else:
            created_rewards = None

        if self.use_custom_fit:
            self.model.fit(formatted_input, formatted_output, batch_size=batch_size,  rewards=created_rewards)
        else:
            self.__fit(formatted_input, formatted_output, batch_size=batch_size, rewards=created_rewards)

    def predict(self, prediction_input):
        """
        Predicts an output given the input
        :param prediction_input: The input, this can be anything as it will go through a BaseInputFormatter
        :return:
        """
        arr = self.input_formatter.create_input_array(prediction_input)
        if self.use_custom_sample_action:
            output = self.model.predict(arr)
        else:
            output = self.__predict(arr)
        return self.output_formatter.format_model_output(output)

    def finish_training(self, save_model=True):
        if save_model:
            file_path = self.get_file_path()
            print('saving model at:', file_path)
            self.model.save(file_path)

    def __fit(self, arr, out, rewards=None, batch_size=1):
        raise NotImplementedError()

    def __predict(self, arr):
        raise NotImplementedError()

    def get_model_name(self):
        return str(type(self.model).__name__)

    def get_file_path(self):
        return get_repo_directory() + '/trainer/weights/' + self.get_model_name() + '.mdl'

    def __load_model_safely(self):
        try:
            self.model.load(self.get_file_path())
        except Exception as e:
            get_logger(str(type(self).__name__)).warn("Unable to load model: " + str(e))
