import abc

from typing import Any, Dict, Union, Collection


class Report(metaclass=abc.ABCMeta):

    _name: str

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> Union[Dict[str, Any], Collection[Dict[str, Any]]]:
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
