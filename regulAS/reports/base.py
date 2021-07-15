import abc


class Report(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        pass
