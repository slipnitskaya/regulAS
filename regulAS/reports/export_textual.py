import os

import pandas as pd

import hydra  # noqa

from regulAS.reports import Report


class ExportCSV(Report):

    def __init__(
        self,
        output_dir: str,
        sep: str = ',',
        decimal: str = '.'
    ):
        super(ExportCSV, self).__init__()

        self.output_dir = output_dir
        self.sep = sep
        self.decimal = decimal

    def generate(self, df: pd.DataFrame) -> None:
        if isinstance(df.index, pd.MultiIndex):
            index = df.index
        else:
            index = df.columns

        data_name, data_md5, *_ = map(lambda x: x.pop(), (set(item) for item in zip(*index)))
        df_title = '-'.join([df.attrs.get('title', ''), self.name, data_name, data_md5])

        if os.path.isabs(self.output_dir):
            output_dir = self.output_dir
        else:
            output_dir = os.path.join(hydra.utils.get_original_cwd(), self.output_dir)

        path_to_output = os.path.abspath(os.path.join(output_dir, f'{df_title}.csv'))
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

        df.to_csv(
            path_to_output,  # noqa
            sep=self.sep,
            decimal=self.decimal
        )
