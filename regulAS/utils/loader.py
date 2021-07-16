import os
import hashlib

import hydra
import pandas as pd

from typing import Tuple, Optional

from regulAS.utils import Loader


class PickleLoader(Loader):

    _data: pd.DataFrame

    def __init__(self,
                 name: str,
                 path_to_file: str,
                 objective: Optional[str] = None,
                 meta: Optional[str] = None):

        super(PickleLoader, self).__init__(name=name, meta=meta)

        self.path_to_file = os.path.join(hydra.utils.to_absolute_path(hydra.utils.get_original_cwd()), path_to_file)
        self.objective = objective

        self._data = self._init_data()

    def _init_data(self) -> pd.DataFrame:
        return pd.read_pickle(self.path_to_file).T

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_col = self.objective or self._data.columns[-1]

        X = self._data.drop(target_col, axis=1)
        y = self._data.loc[:, [target_col]]

        return X, y

    @property
    def name(self):
        return f'{super(PickleLoader, self).name}-{os.path.basename(self.path_to_file)}'

    @property
    def num_samples(self) -> int:
        return self._data.shape[0]

    @property
    def num_features(self) -> int:
        return self._data.shape[1] - 1

    @property
    def md5(self) -> str:
        md5 = hashlib.md5()

        with open(self.path_to_file, 'rb') as pkl:
            for chunk in iter(lambda: pkl.read(4096), b''):
                md5.update(chunk)

        md5.update(self._name.encode())
        md5.update(self._meta.encode())

        return md5.hexdigest()
