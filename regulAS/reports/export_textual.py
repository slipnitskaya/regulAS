import os

import pandas as pd

import hydra

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

    def generate(self, *dataframes) -> None:
        df: pd.DataFrame
        for df in dataframes:
            path_to_output = os.path.abspath(os.path.join(
                hydra.utils.get_original_cwd(),
                os.path.basename(self.output_dir),
                f'{df.attrs["title"]}.csv'
            ))

            os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

            df.to_csv(
                path_to_output,
                sep=self.sep,
                decimal=self.decimal
            )
