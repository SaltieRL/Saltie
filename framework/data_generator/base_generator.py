from rlbot.utils.logging_utils import get_logger


class BaseDataGenerator:
    def __init__(self):
        self.logger = get_logger(type(self).__name__)
        pass

    def initialize(self, **kwargs):
        raise NotImplementedError()

    def get_data(self):
        """
        Gets all the data needed for a model.
        :return: The result can be used as an iterator.
        """
        while self.has_next():
            yield self._next()

    def has_next(self):
        raise NotImplementedError()

    def _next(self):
        raise NotImplementedError()
