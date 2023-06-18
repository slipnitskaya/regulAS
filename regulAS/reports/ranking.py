import os
import difflib
import itertools

import numpy as np
import pandas as pd

import regulAS.persistence as persistence

from collections import defaultdict

from pyensembl import EnsemblRelease
from sqlalchemy import and_, func

from typing import Dict, List, Tuple, Optional

from regulAS.core import RegulAS
from regulAS.reports import Report


os.environ['PYENSEMBL_CACHE_DIR'] = os.path.expanduser('~/.local/cache/pyensembl')


class FeatureRankingReport(Report):

    ENSEMBLE_RELEASE: int = 102
    SPECIES: str = 'homo_sapiens'

    def __init__(
        self,
        experiment_name: str,
        sort_by: str,
        sort_ascending: bool = True,
        top_k_models: Optional[int] = None,
        top_k_features: Optional[int] = None,
        symbolic_aliases: bool = True
    ):
        super(FeatureRankingReport, self).__init__()

        self.experiment_name = experiment_name
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.top_k_models = top_k_models
        self.top_k_features = top_k_features
        self.symbolic_aliases = symbolic_aliases

        if self.symbolic_aliases:
            self._ensembl = EnsemblRelease(release=self.ENSEMBLE_RELEASE, species=self.SPECIES)
            self._ensembl.download()
            self._ensembl.index()

    def generate(self, df: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        conn = RegulAS().db_connection

        (
            data_name,
            data_md5,
            experiment_name,
            experiment_md5,
            *_
        ) = map(lambda x: x.pop(), (set(item) for item in zip(*df.index)))

        perf_groups = df.droplevel(['data', 'data_md5', 'experiment_name', 'experiment_md5']).groupby(level='model')
        sort_col = difflib.get_close_matches(self.sort_by, df.columns, n=1)

        top_k_models = max(0, self.top_k_models) or perf_groups.ngroups

        result = defaultdict(  # data name ->
            lambda: defaultdict(  # data MD5 ->
                lambda: defaultdict(  # experiment name ->
                    lambda: defaultdict(  # experiment MD5 ->
                        lambda: defaultdict(  # model name ->
                            lambda: defaultdict(  # hyper-parameters MD5 ->
                                lambda: defaultdict(  # fold ->
                                    dict  # feature name -> feature score
                                )
                            )
                        )
                    )
                )
            )
        )

        perf_groups = sorted(
            map(lambda x: (x[0], x[1].sort_values(by=sort_col, ascending=self.sort_ascending)), perf_groups),
            key=lambda x: next(x[1].iterrows())[1][sort_col].item(),
            reverse=not self.sort_ascending
        )

        model_aliases = dict()
        columns_score = set()
        num_models_chosen = 0
        for model_name, model_df in perf_groups:
            hyper_parameters_md5, model_scores = next(model_df.droplevel('model').iterrows())

            pipelines = conn.query(
                persistence.Pipeline
            ).join(
                persistence.Experiment,
                persistence.Data,
                persistence.TransformationSequence,
                persistence.Transformation,
            ).filter(
                and_(
                    persistence.Data.md5 == data_md5,
                    persistence.Experiment.md5 == experiment_md5,
                    persistence.TransformationSequence.md5 == hyper_parameters_md5,
                    persistence.Transformation.fqn == model_name
                )
            )

            num_rankings_found = 0
            for pipeline in pipelines:
                ranking = conn.query(
                    persistence.FeatureRanking.feature,
                    persistence.FeatureRanking.score
                ).filter(
                    persistence.FeatureRanking.pipeline == pipeline
                ).order_by(
                    func.abs(persistence.FeatureRanking.score).desc()
                )

                if ranking is not None:
                    alias, = conn.query(
                        persistence.TransformationSequence.alias
                    ).join(
                        persistence.Transformation,
                        persistence.Pipeline
                    ).filter(
                        and_(
                            persistence.Transformation.fqn == model_name,
                            persistence.Pipeline.idx == pipeline.idx
                        )
                    ).first()

                    model_aliases[(model_name, hyper_parameters_md5)] = alias

                    if self.top_k_features is not None:
                        ranking = ranking.limit(self.top_k_features)

                    column_score = f'score:{pipeline.fold}'
                    for feature, score in ranking:
                        if self.symbolic_aliases:
                            try:
                                feature = self._ensembl.gene_name_of_gene_id(feature)
                            except ValueError:
                                pass

                        result[data_name][data_md5][experiment_name][experiment_md5][model_name][
                            hyper_parameters_md5
                        ][column_score][feature] = score

                        result[data_name][data_md5][experiment_name][experiment_md5][model_name][
                            hyper_parameters_md5
                        ]['score:mean'][feature] = np.nan

                        result[data_name][data_md5][experiment_name][experiment_md5][model_name][
                            hyper_parameters_md5
                        ]['score:std'][feature] = np.nan

                    columns_score.add(column_score)
                    num_rankings_found += 1

            if num_rankings_found > 0:
                num_models_chosen += 1

            if num_models_chosen >= top_k_models:
                break

        result = pd.DataFrame.from_dict(
            {
                (
                    data_name,
                    data_md5,
                    experiment_name,
                    experiment_md5,
                    model_name,
                    hyper_parameters_md5,
                    column_score
                ): result[data_name][data_md5][experiment_name][
                    experiment_md5
                ][model_name][hyper_parameters_md5][column_score]

                for data_name in result.keys()
                for data_md5 in result[data_name].keys()
                for experiment_name in result[data_name][data_md5].keys()
                for experiment_md5 in result[data_name][data_md5][experiment_name].keys()
                for model_name in result[data_name][data_md5][experiment_name][experiment_md5].keys()
                for hyper_parameters_md5 in result[data_name][data_md5][experiment_name][
                    experiment_md5
                ][model_name].keys()
                for column_score in result[data_name][data_md5][experiment_name][
                    experiment_md5
                ][model_name][hyper_parameters_md5].keys()
            },
            orient='columns'
        )
        result.attrs['title'] = '-'.join([df.attrs.get('title', ''), self.name])
        result.attrs['model_aliases'] = model_aliases
        result.index.set_names(['feature'], inplace=True)
        result.columns.set_names(
            ['data', 'data_md5', 'experiment_name', 'experiment_md5', 'model', 'hyper_parameters_md5', 'score'],
            inplace=True
        )

        column_groups = list(result.groupby(axis=1, level=result.columns.names[:-1]).groups)
        column_group: Tuple
        for column_group in column_groups:
            columns = list(
                map(
                    lambda x: x[0] + x[1],
                    itertools.product([column_group], map(lambda col: (col,), columns_score))
                )
            )

            scores = result[columns]
            result[column_group + ('score:mean',)] = np.mean(scores, axis=1)
            result[column_group + ('score:std',)] = np.std(scores, axis=1)

            result = result.drop(columns=columns)

        dataframes = list()
        for data_name, df in result.groupby(axis=1, level='data'):
            dataframes.append({'df': df.sort_index()})

        return dataframes
