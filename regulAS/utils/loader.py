import os
import sys
import glob
import gzip
import hashlib
import logging
import argparse
import requests
import multiprocessing as mp

import tqdm
import hydra  # noqa
import numpy as np
import pandas as pd

from enum import auto, Enum
from typing import List, Tuple, Optional

from regulAS.utils import Loader


class ArgTypeMixin(Enum):

    @classmethod
    def argtype(cls, s: str) -> Enum:
        try:
            return cls[s]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"{s!r} is not a valid {cls.__name__}")

    def __str__(self):
        return self.name


class Cohort(ArgTypeMixin, Enum):

    TCGA = auto()
    GTEX = auto()


class Condition(ArgTypeMixin, Enum):

    Tumor = auto()
    Normal = auto()
    Combined = auto()


class CsvGzReader(object):

    def __init__(self, sep: str = ';'):
        super(CsvGzReader, self).__init__()
        self.sep = sep

    @staticmethod
    def _read_gz(path_to_gz: str) -> Tuple[int, str]:
        with gzip.open(path_to_gz, 'r') as gz:
            gz = filter(lambda l: l[:6] != b'TARGET', gz)
            for idx, line in enumerate(gz):
                yield idx, line.decode().strip('\r\n')

    @staticmethod
    def _parse_worker(args) -> Tuple[int, str, np.ndarray]:
        idx, line, sep = args

        line_idx, *cols = line.split(sep)
        try:
            cols = np.array([float(val) for val in cols], dtype=np.float32)
        except ValueError:
            cols = np.array(cols)

        return idx, line_idx, cols

    def read(self, path_to_gz: str) -> pd.DataFrame:
        csv = map(lambda x: (*x, self.sep), iter(self._read_gz(path_to_gz)))
        _, index_name, columns = self._parse_worker(next(csv))

        rows = list()
        indices = list()
        with mp.Pool(mp.cpu_count()) as pool:
            for idx, line_idx, cols in pool.imap(self._parse_worker, csv, chunksize=500):
                rows.append(cols)
                indices.append(line_idx)

        df = pd.DataFrame(index=indices, columns=columns, dtype=np.float32)
        for idx, line_idx in enumerate(tqdm.tqdm(indices, desc='Loading data')):
            df.loc[line_idx] = rows[idx]

        df.columns.set_names([index_name], inplace=True)

        return df


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
        return pd.read_pickle(self.path_to_file)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_col = self.objective or self._data.columns[-1]

        X = self._data.drop(target_col, axis=1)  # noqa
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


class RawLoader(PickleLoader):

    def __init__(self,
                 name: str,
                 gene_symbol: str,
                 dir_to_data: str = 'data_reference',
                 out_dir: str = 'data',
                 condition: str = 'Tumor',
                 cutoff_reads: int = 5,
                 ratio_nan_to_drop: float = 1.0,
                 log_transform: bool = False,
                 sample_size_per_tissue_min: int = 50,
                 pair_tissues: bool = True,
                 include_adjacent_tissues: bool = False,
                 objective: Optional[str] = None,
                 meta: Optional[str] = None):

        self.gene_symbol = gene_symbol
        self.dir_to_data = os.path.join(hydra.utils.to_absolute_path(hydra.utils.get_original_cwd()), dir_to_data)
        self.out_dir = os.path.join(hydra.utils.to_absolute_path(hydra.utils.get_original_cwd()), out_dir)
        self.condition = Condition[condition]
        self.cutoff_reads = cutoff_reads
        self.ratio_nan_to_drop = ratio_nan_to_drop
        self.log_transform = log_transform
        self.sample_size_per_tissue_min = sample_size_per_tissue_min
        self.pair_tissues = pair_tissues
        self.include_adjacent_tissues = include_adjacent_tissues

        super(RawLoader, self).__init__(
            name=name,
            path_to_file='',
            objective=objective,
            meta=meta
        )

    def _init_data(self) -> pd.DataFrame:
        url_to_rnaseq = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz'
        url_to_metadata = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz'
        dir_to_junctions = os.path.join(self.dir_to_data, 'junctions')
        path_to_gene_ids = os.path.join(self.dir_to_data, 'gene_ids.txt')

        self.log(logging.INFO, f'Loading RNA-Seq and Junction data...')
        # load gene expression
        rnaseq = self.load_from_xena(url_to_rnaseq, self.dir_to_data)

        # load metadata
        metadata = self.load_from_xena(url_to_metadata, self.dir_to_data)

        # subset tumor and normal samples
        tcga_samples_to_subset: List[str] = ['Primary Tumor']

        if self.include_adjacent_tissues:
            # include adjacent normal tissues
            tcga_samples_to_subset += ['Solid Tissue Normal']

        cond_comb = (
            (metadata['_study'] == Cohort.GTEX.name) & (metadata['_sample_type'].isin(['Normal Tissue']))
        ) | (
            (metadata['_study'] == Cohort.TCGA.name) & (metadata['_sample_type'].isin(tcga_samples_to_subset))
        )
        metadata = metadata[cond_comb]

        # rename columns
        cols = {'_primary_site': 'tissue', '_study': 'cohort'}
        metadata = metadata[list(cols.keys())].rename(columns=cols)

        # load junctions
        try:
            path_to_junctions = glob.glob(f'{dir_to_junctions}/{self.gene_symbol}*.csv')
            junctions = pd.read_csv(path_to_junctions[0], sep=',', index_col=0)
        except (FileNotFoundError, ValueError, IndexError) as ex:
            raise ex.__class__(f'Junction data are not found for {self.gene_symbol}.') from ex

        # load genes to subset
        gene_list = open(path_to_gene_ids, 'r').readlines()
        gene_list = list(map(lambda l: l.strip('\n'), gene_list))

        self.log(logging.INFO, f'Processing {self.gene_symbol} {self.condition.name.lower()} data...')
        # process junctions
        psi = self.process_junctions(
            junctions=junctions,
            cutoff_reads=self.cutoff_reads
        )

        # transpose data (samples x features)
        rnaseq = rnaseq.T
        psi = psi.T

        # process RNA-Seq
        rnaseq = self.process_rnaseq(
            rnaseq=rnaseq,
            gene_ids=gene_list,
            ratio_nan_to_drop=self.ratio_nan_to_drop,
            log_transform=self.log_transform
        )

        # combine data
        data = self.prepare_data(
            rnaseq=rnaseq,
            psi=psi,
            metadata=metadata,
            condition=self.condition,
            pair_tissues=self.pair_tissues,
            sample_size_per_tissue_min=self.sample_size_per_tissue_min
        )

        self.log(logging.INFO, f'Data is processed for {data.shape[0]} samples and {data.shape[1] - 1} genes.')

        # save data
        setup_id = self.get_setup_id(
            condition=self.condition,
            cutoff_reads=self.cutoff_reads,
            log_transform=self.log_transform,
            pair_tissues=self.pair_tissues,
            include_adjacent_tissues=self.include_adjacent_tissues,
            sample_size_per_tissue_min=self.sample_size_per_tissue_min,
        )
        filename = f'{self.gene_symbol}_{setup_id}.pkl'
        path_to_data = os.path.join(self.out_dir, self.gene_symbol, filename)
        os.makedirs(os.path.dirname(path_to_data), exist_ok=True)

        self.log(logging.INFO, f'Saving data as `{filename}`...')
        pd.to_pickle(data, path_to_data)
        self.path_to_file = path_to_data

        return data

    @staticmethod
    def load_from_xena(url_to_data: str, dir_to_data: str) -> pd.DataFrame:
        path_to_local = os.path.join(dir_to_data, os.path.basename(url_to_data))
        if not os.path.isfile(path_to_local):
            response = requests.get(url_to_data, stream=True)
            with open(path_to_local, 'wb') as df_local:
                for chunk in response.iter_content(chunk_size=1024):
                    df_local.write(chunk)
        path_to_local_df = f'{path_to_local}.pkl'
        if os.path.exists(path_to_local_df):
            df = pd.read_pickle(path_to_local_df)
        else:
            df = CsvGzReader(sep='\t').read(path_to_local)
            pd.to_pickle(df, path_to_local_df)

        return df

    @staticmethod
    def calculate_psi(reads: pd.DataFrame) -> pd.DataFrame:

        inclusion = reads.filter(regex='^inclusion', axis=0).sum(axis=0).values / 2
        exclusion = reads.filter(regex='^exclusion', axis=0).values

        psi = (inclusion / (inclusion + exclusion))

        sample_ids = reads.columns.values.tolist()
        psi = pd.DataFrame.from_records(psi, columns=sample_ids, index=['psi'])

        return psi

    @classmethod
    def filter_reads(
            cls,
            reads: pd.DataFrame,
            cutoff_reads: float
    ) -> pd.DataFrame:

        # subset samples w.r.t. coverage of reads
        if cutoff_reads == 0:
            # remove samples with no reads (zeros in all junctions)
            samples_coverage_low = reads.columns[(reads == cutoff_reads).all().values].values.tolist()
        else:
            # remove samples w.r.t. the defined cutoff
            samples_coverage_low = reads.columns[(reads < cutoff_reads).all().values].values.tolist()

        if len(samples_coverage_low) > 0:
            cls.log(logging.INFO,
                          f'\tFiltering {len(samples_coverage_low)} low-coverage (cutoff={cutoff_reads}) samples...'
                          )

        reads = reads.drop(samples_coverage_low, axis=1)

        if reads.empty:
            sys.exit(
                'Program is terminated as no samples left in filtered reads data (all samples have low-coverage).'
                'Try to update `cutoff_reads` parameter (minimize it or pass `None` to ignore the filtering)'
                'and/or include more samples and/or analyse another data.')
        else:
            return reads

    @classmethod
    def process_junctions(
            cls,
            junctions: pd.DataFrame,
            cutoff_reads: Optional[int] = 0
    ) -> pd.DataFrame:

        junctions = junctions.fillna(0)

        # remove samples w.r.t. read coverage
        if cutoff_reads >= 0:
            junctions = cls.filter_reads(junctions, cutoff_reads)

        # calculate PSI
        psi = cls.calculate_psi(junctions)
        psi.index = ['psi']

        # remove samples with missing values
        psi = psi.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        return psi

    @classmethod
    def process_rnaseq(
            cls,
            rnaseq: pd.DataFrame,
            gene_ids: List,
            ratio_nan_to_drop: Optional[float] = 1.0,
            log_transform: bool = False
    ) -> pd.DataFrame:

        # filter genes w.r.t. missing values
        mask_sparse: np.array = rnaseq.isna().values
        if mask_sparse.any():
            if ratio_nan_to_drop is not None:
                if ratio_nan_to_drop == 1.0:
                    mask_sparse = mask_sparse.any(axis=0)
                else:
                    mask_sparse = mask_sparse.mean(axis=0) > ratio_nan_to_drop

                n_rbps_sparse = mask_sparse.sum()
                genes_sparse = rnaseq.columns[mask_sparse]
                rnaseq = rnaseq.drop(genes_sparse, axis=1)

                cls.log(
                    logging.INFO,
                    '\tFiltering {} features (sparsity={:.1%})...'.format(n_rbps_sparse, 1.0 - ratio_nan_to_drop)
                )
            if ratio_nan_to_drop != 1.0:
                rnaseq = rnaseq.fillna(0)

        # subset genes
        genes_found = list(set(gene_ids) & set(rnaseq.columns))
        if not genes_found:
            # format gene names to match
            sep_gene_ids = '.'
            gene_ids_all = rnaseq.columns.to_list()
            if any([sep_gene_ids in idx for idx in gene_ids]) or any([sep_gene_ids in idx for idx in gene_ids_all]):
                gene_ids = [idx.split(sep_gene_ids)[0] for idx in gene_ids]
                rnaseq.columns = [idx.split(sep_gene_ids)[0] for idx in gene_ids_all]
            genes_found = list(set(gene_ids) & set(rnaseq.columns))
        if genes_found:
            # subset genes
            cls.log(logging.INFO, f'\tSelecting {len(genes_found)} genes...')
            rnaseq = rnaseq.reindex(genes_found, axis=1)
        else:
            sys.exit('Program is terminated as no gene IDs to subset are found in the RNA-Seq data.'
                     'Pass a list of another genes IDs as `gene_ids` and/or check whether Ensembl IDs are correct.')

        # log-transform values
        if log_transform:
            cls.log(logging.INFO, '\tLog-transforming expression values...')
            rnaseq = rnaseq.apply(lambda x: np.log2(x + 1e-3))

            if rnaseq.isnull().sum().sum() > 0:
                cls.log(logging.WARNING, 'Log-transformed data contain NaN values')

        return rnaseq

    @classmethod
    def prepare_data(
            cls,
            rnaseq: pd.DataFrame,
            psi: pd.DataFrame,
            metadata: pd.DataFrame,
            condition: Condition,
            pair_tissues: Optional[bool] = None,
            sample_size_per_tissue_min: Optional[int] = None
    ) -> pd.DataFrame:

        # aggregate data
        data = rnaseq.join(psi, how='inner')

        # annotate data
        data = data.join(metadata, how='inner')

        # subset conditions and tissues
        data = cls.subset_conditions(
            data, condition=condition, pair_tissues=pair_tissues, sample_size_per_tissue_min=sample_size_per_tissue_min
        )

        # remove annotation
        data = data.select_dtypes(include=[np.number])

        return data

    @classmethod
    def subset_conditions(
            cls,
            data_annotated: pd.DataFrame,
            condition: Condition,
            pair_tissues: Optional[bool] = False,
            sample_size_per_tissue_min: Optional[bool] = False
    ) -> pd.DataFrame:

        # pair tissues in TCGA and GTEx data
        if pair_tissues:
            tissues_common = list(
                set(data_annotated.loc[data_annotated['cohort'] == Cohort.TCGA.name, 'tissue']).intersection(
                    set(data_annotated.loc[data_annotated['cohort'] == Cohort.GTEX.name, 'tissue'])))
            data_annotated = data_annotated[data_annotated['tissue'].isin(tissues_common)]
            cls.log(logging.INFO, f"\tSelecting {data_annotated['tissue'].nunique()} common tissues...")

        # subset conditions
        if condition != Condition.Combined:
            try:
                cohort_name: str = Cohort.TCGA.name if condition == Condition.Tumor else Cohort.GTEX.name
                mask_subsets = data_annotated['cohort'].str.contains(cohort_name)
                data_annotated = data_annotated[mask_subsets]
            except ValueError:
                raise logging.warning(
                    f'`{condition.name}` samples are not found. Check out data or/and Try another condition(s).')

        # subset tissues by sample size
        samples_per_tissue = data_annotated['tissue'].value_counts()
        cond_tissues_with_small_sample_size = samples_per_tissue < sample_size_per_tissue_min
        if sample_size_per_tissue_min is not None and np.any(cond_tissues_with_small_sample_size):
            tissues_to_be_filtered_out = samples_per_tissue[cond_tissues_with_small_sample_size].index.to_list()
            data_annotated = data_annotated.loc[~data_annotated['tissue'].isin(tissues_to_be_filtered_out), :]
            cls.log(
                logging.INFO,
                f"\tSelecting {data_annotated['tissue'].nunique()} tissues with at least {sample_size_per_tissue_min} samples..."
            )

        return data_annotated

    @staticmethod
    def get_setup_id(
            condition: Condition,
            log_transform: bool,
            pair_tissues: bool,
            include_adjacent_tissues: bool,
            cutoff_reads: Optional[int] = None,
            sample_size_per_tissue_min: Optional[int] = None,
    ) -> str:

        setup_id = f'r{cutoff_reads if cutoff_reads >= 0 else ""}'
        setup_id += 'Log' if log_transform else ''

        tissue_selection = ''
        tissue_selection += 'A' if include_adjacent_tissues else ''
        tissue_selection += 'P' if pair_tissues else ''
        setup_id += f't{tissue_selection}' if tissue_selection else ''

        if sample_size_per_tissue_min is not None and sample_size_per_tissue_min > 0:
            setup_id += f's{sample_size_per_tissue_min}'

        setup_id += f'c{condition.name}'

        return setup_id

    @property
    def meta(self):
        return str({
            'gene_symbol': self.gene_symbol,
            'dir_to_data': self.dir_to_data,
            'out_dir': self.out_dir,
            'condition': self.condition,
            'cutoff_reads': self.cutoff_reads,
            'ratio_nan_to_drop': self.ratio_nan_to_drop,
            'log_transform': self.log_transform,
            'pair_tissues': self.pair_tissues,
            'include_adjacent_tissues': self.include_adjacent_tissues,
            'sample_size_per_tissue_min': self.sample_size_per_tissue_min,
            'meta': super(RawLoader, self).meta
        })
