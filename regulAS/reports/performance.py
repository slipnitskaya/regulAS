import numpy as np
import pandas as pd

import hydra

import regulAS.persistence as persistence

from collections import defaultdict

from sqlalchemy import and_

from typing import Callable, Iterable, Optional

from regulAS.core import RegulAS
from regulAS.reports import Report
from regulAS.utils import load_ndarray


class ModelPerformanceReport(Report):

    metric: Callable[[Iterable[float], Iterable[float]], float]
    greater_is_better: bool

    def __init__(
        self,
        experiment_name: str,
        score_fn: Optional[str] = None,
        loss_fn: Optional[str] = None
    ):
        super(ModelPerformanceReport, self).__init__()

        self.experiment_name = experiment_name
        if score_fn is not None:
            self.metric = hydra.utils.get_method(score_fn)
            self.greater_is_better = True
        elif loss_fn is not None:
            self.metric = hydra.utils.get_method(loss_fn)
            self.greater_is_better = False
        else:
            raise ValueError(
                'Metric function is missing. Please pass a value to either`score_fn` or `loss_fn` argument.'
            )

    def generate(self) -> pd.DataFrame:
        metric_name = f'{self.metric.__name__}:{{}}:{{}}'

        conn = RegulAS().db_connection

        pipelines = conn.query(
            persistence.Pipeline
        ).filter(
            and_(
                persistence.Experiment.name == self.experiment_name,
                persistence.Experiment.idx == persistence.Pipeline.experiment_idx,
                persistence.Pipeline.success
            )
        )

        folds = set(map(lambda row: row[0], pipelines.with_entities(persistence.Pipeline.fold)))
        y_true_test, y_pred_test = zip(*[(list(), list()) for _ in range(len(folds))])
        y_true_train, y_pred_train = zip(*[(list(), list()) for _ in range(len(folds))])

        result = defaultdict(lambda: defaultdict(defaultdict))

        columns_test, columns_train = set(), set()

        pipeline: persistence.Pipeline
        for pipeline in pipelines:
            data_name = f'{pipeline.experiment.data.name}:{pipeline.experiment.data.md5[-6:]}'

            model = conn.query(
                persistence.Transformation
            ).join(
                persistence.Transformation.pipelines, persistence.Pipeline
            ).filter(
                and_(
                    persistence.Pipeline.idx == pipeline.idx,
                    persistence.Transformation.type_ == persistence.Transformation.Type.MODEL
                )
            ).first()

            hyper_parameters = conn.query(
                persistence.HyperParameter.name,
                persistence.HyperParameterValue.value
            ).join(
                persistence.HyperParameterValue,
                persistence.TransformationSequence,
                persistence.Transformation,
                persistence.Pipeline
            ).filter(
                and_(
                    persistence.Transformation.idx == model.idx,
                    persistence.Pipeline.idx == pipeline.idx
                )
            )

            hyper_parameters_md5 = conn.query(
                persistence.TransformationSequence
            ).join(
                persistence.Transformation,
                persistence.Pipeline
            ).filter(
                and_(
                    persistence.Transformation.idx == model.idx,
                    persistence.Pipeline.idx == pipeline.idx
                )
            ).first().md5

            model_hyper_parameters = list()
            for name, value in hyper_parameters:
                model_hyper_parameters.append((name, value))

            model_hyper_parameters = ', '.join([
                f'{name}: {value}' for name, value in sorted(model_hyper_parameters, key=lambda x: x[0])
            ])
            model_hyper_parameters = f'{{{model_hyper_parameters}}}'

            model_name = f'{model.fqn}:{hyper_parameters_md5[-6:]}'

            predictions = conn.query(
                persistence.Prediction
            ).filter(
                persistence.Prediction.pipeline == pipeline
            )

            prediction: persistence.Prediction
            for prediction in predictions:
                if prediction.training:
                    y_true_train[pipeline.fold].append(load_ndarray(prediction.true_value).astype(np.float16))
                    y_pred_train[pipeline.fold].append(load_ndarray(prediction.predicted_value).astype(np.float16))
                else:
                    y_true_test[pipeline.fold].append(load_ndarray(prediction.true_value).astype(np.float16))
                    y_pred_test[pipeline.fold].append(load_ndarray(prediction.predicted_value).astype(np.float16))

            metric_test = self.metric(
                np.array(y_true_test[pipeline.fold]).squeeze(),
                np.array(y_pred_test[pipeline.fold]).squeeze()
            )
            try:
                metric_test = next(iter(metric_test))
            except TypeError:
                pass

            metric_train = self.metric(
                np.array(y_true_train[pipeline.fold]).squeeze(),
                np.array(y_pred_train[pipeline.fold]).squeeze()
            )
            try:
                metric_train = next(iter(metric_train))
            except TypeError:
                pass

            column_test = metric_name.format('test', pipeline.fold)
            column_train = metric_name.format('train', pipeline.fold)

            result[data_name][model_name]['hyper_parameters'] = model_hyper_parameters
            result[data_name][model_name][column_test] = metric_test
            result[data_name][model_name][column_train] = metric_train

            columns_test.add(column_test)
            columns_train.add(column_train)

        result = pd.DataFrame.from_dict({
            (data_name, model_name): result[data_name][model_name]
            for data_name in result.keys()
            for model_name in result[data_name].keys()
        }, orient='index')
        result.index.set_names(['data', 'model'])
        result.attrs['title'] = self.experiment_name

        for model_name in result.index:
            scores_test, scores_train = result.loc[model_name, columns_test], result.loc[model_name, columns_train]

            mean_test, std_test = np.mean(scores_test), np.std(scores_test)
            mean_train, std_train = np.mean(scores_train), np.std(scores_train)

            result.loc[model_name, metric_name.format('test', 'mean')] = mean_test
            result.loc[model_name, metric_name.format('test', 'std')] = std_test
            result.loc[model_name, metric_name.format('train', 'mean')] = mean_train
            result.loc[model_name, metric_name.format('train', 'std')] = std_train

        return result.drop(
            columns=columns_test | columns_train
        ).sort_values(
            by=[metric_name.format('test', 'mean'), metric_name.format('test', 'std')],
            ascending=not self.greater_is_better
        )
