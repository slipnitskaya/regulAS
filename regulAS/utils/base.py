import abc

import pandas as pd

from typing import Tuple, Optional


class Loader(metaclass=abc.ABCMeta):

    def __init__(self, name: str, meta: Optional[str] = None):
        super(Loader, self).__init__()

        self._name = name
        self._meta = meta

    @property
    def name(self):
        return self._name

    @property
    def meta(self):
        return self._meta

    @abc.abstractmethod
    def load(self) -> Tuple[pd.DataFrame, ...]:
        pass

    @property
    @abc.abstractmethod
    def num_samples(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_features(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def md5(self) -> str:
        pass


class Split(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def split(self, X, y=None, groups=None):
        pass

    @property
    @abc.abstractmethod
    def n_splits(self) -> int:
        pass
