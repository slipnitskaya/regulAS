import abc
import logging

import numpy as np
import pandas as pd

import msgpack
import msgpack_numpy as m

from typing import Tuple, Optional


m.patch()


class Loader(metaclass=abc.ABCMeta):

    log = staticmethod(logging.log)

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


def dump_ndarray(data: np.ndarray) -> bytes:
    return msgpack.packb(data)


def load_ndarray(data: bytes) -> np.ndarray:
    return msgpack.unpackb(data)
