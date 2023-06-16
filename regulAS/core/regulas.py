import os
import sys
import enum
import random
import hashlib
import inspect
import logging
import traceback
import networkx as nx
import multiprocessing as mp

import hydra  # noqa

import numpy as np
import pandas as pd

from collections import defaultdict

from typing import cast, Any, Dict, List, Type, Tuple, Union, Optional, Collection

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig  # noqa

from sqlalchemy import and_, create_engine

from sqlalchemy.orm import Session, Query
from sqlalchemy.engine.url import URL, make_url

from regulAS import persistence
from regulAS.utils import Loader, Split, dump_ndarray


SAMPLES: Optional[pd.DataFrame] = None
TARGETS: Optional[pd.DataFrame] = None
TRAIN_IDS: Optional[List[np.ndarray]] = None
TEST_IDS: Optional[List[np.ndarray]] = None


class Singleton(type):

    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class RegulAS(metaclass=Singleton):

    SCORE_ATTR_NAMES = ['coef_', 'feature_importances_']

    _logger: logging.Logger = logging.getLogger(__name__.split('.')[0])

    _num_processes: int
    _session: Session
    _dataset: Loader
    _splitter: Split

    _samples: pd.DataFrame
    _targets: pd.DataFrame
    _other: Optional[Collection[Any]]

    _train_ids: List[np.ndarray]
    _test_ids: List[np.ndarray]

    _tasks: [int, Union[Tuple[int, DictConfig, np.ndarray, np.ndarray], Dict[str, Any]]]
    _num_tasks: int
    _num_finished: int

    _experiment: persistence.Experiment

    class InitStatus(enum.Flag):
        NONE = enum.auto()
        ALLOW_SUBMIT = enum.auto()
        ALLOW_REPORTS = enum.auto()
        FAILED = enum.auto()

    @property
    def db_connection(self):
        return self._session

    def init(
        self,
        cfg: DictConfig
    ) -> InitStatus:

        self.seed_everything(cfg.random_state)

        self._tasks = dict()
        self._num_processes = max(cfg.num_processes, 0) or mp.cpu_count()

        db_url: URL = make_url(cfg.database.url)
        db_url = URL.create(
            drivername=db_url.drivername,
            username=db_url.username,
            password=db_url.password,
            host=db_url.host,
            port=db_url.port,
            database=os.path.join(hydra.utils.to_absolute_path(hydra.utils.get_original_cwd()), db_url.database),
            query=db_url.query
        )
        db_url = db_url.update_query_dict({'check_same_thread': 'false'}, append=True)
        self._session = Session(bind=create_engine(db_url))

        persistence.create_schema(db_url, self._session.bind)

        init_status = self.InitStatus.NONE

        data = self._init_data(cfg)

        if cfg.experiment.pipelines:
            self._train_ids, self._test_ids = list(), list()
            for train_ids, test_ids in self._splitter.split(self._samples, self._targets, *self._other):
                self._train_ids.append(train_ids)
                self._test_ids.append(test_ids)

            cfg_str = OmegaConf.to_yaml(cfg)
            experiment_md5 = hashlib.md5()
            experiment_md5.update(cfg_str.encode())
            experiment_md5.update(data.md5.encode())
            experiment_md5 = experiment_md5.hexdigest()

            if self.db_connection.query(
                    persistence.Experiment
            ).filter(
                persistence.Experiment.md5 == experiment_md5
            ).first() is not None:
                self.log(
                    logging.WARNING,
                    f'Experiment "{cfg.experiment.name}" (unique ID, MD5): {experiment_md5}) '
                    'is already present in the database. Skipping.'
                )

                init_status = self.InitStatus.FAILED
            else:
                self._experiment = persistence.Experiment(
                    name=cfg.experiment.name,
                    data=data,
                    config=cfg_str,
                    md5=experiment_md5,
                    random_seed=cfg.random_state
                )

                self.db_connection.add(self._experiment)
                self.db_connection.commit()

                init_status |= self.InitStatus.ALLOW_SUBMIT

        if cfg.experiment.reports:
            init_status |= self.InitStatus.ALLOW_REPORTS

        return init_status

    @staticmethod
    def seed_everything(seed: Optional[int]) -> None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def log(level, msg, *args, **kwargs):
        RegulAS._logger.log(level, msg, *args, **kwargs)

    def submit(
        self,
        tasks: Collection[DictConfig]
    ) -> None:

        num_tasks = len(tasks)
        num_tasks_total = num_tasks * self._splitter.n_splits
        num_processes = min(self._num_processes, num_tasks_total)
        self.log(
            logging.INFO,
            f'{num_tasks} tasks were prepared for '
            f'{self._splitter.n_splits}-fold cross-validation '
            f'using {num_processes} workers '
            f'({num_tasks_total} tasks overall)'
        )

        task_idx = 1
        for task in tasks:
            for fold in range(self._splitter.n_splits):
                self._tasks[task_idx] = task_idx, task, fold
                task_idx += 1

        with mp.Pool(
            processes=num_processes,
            initializer=self._init_pool,
            initargs=(self._samples, self._targets, self._train_ids, self._test_ids)
        ) as pool:

            self._num_tasks = len(self._tasks)
            self._num_finished = 0

            for idx, task in self._tasks.items():
                pool.apply_async(func=self._run_task, args=task, callback=self._process_task_result)
                self.log(logging.INFO, f'Task {idx}/{self._num_tasks} has been submitted.')

            pool.close()
            pool.join()

    def generate(
        self,
        report_tasks: DictConfig
    ) -> None:

        dependencies = defaultdict(list)
        for report_name, report_cfg in report_tasks.items():
            report_dependencies = report_cfg.get('_depends_on_', dict())
            dependencies[report_name].extend(report_dependencies[name] for name in sorted(report_dependencies.keys()))

        for report_name in dependencies.keys():
            dependencies[report_name] = list(dict.fromkeys(dependencies[report_name]))

        src, dst = set(dependencies.keys()), {dst for dep in dependencies.values() for dst in dep}
        missing_deps = dst - src
        if missing_deps:
            RegulAS.log(logging.ERROR, f'Missing report dependencies detected: {missing_deps}. Stopping.')
            return

        dependency_graph = [(src, dst) for src in dependencies.keys() for dst in dependencies[src] if dst]
        cycles = list(nx.simple_cycles(nx.DiGraph(dependency_graph)))
        if cycles:
            RegulAS.log(logging.ERROR, f'Circular report dependencies detected: {cycles}. Stopping.')
            return

        dependencies = [(src, dst) for src, dst in dependencies.items()]
        resolution_order = list()
        while dependencies:
            src, dst = dependencies.pop(0)
            if not dst or set(dst).issubset(set(resolution_order)):
                resolution_order.append(src)
            else:
                dependencies.append((src, dst))

        generated = defaultdict(list)
        for report_name in resolution_order:
            self.log(logging.INFO, f'Generating "{report_name}"...')

            try:
                report_cfg = report_tasks[report_name]

                report_dependencies = report_cfg.pop('_depends_on_', dict())
                num_deps = len(report_dependencies)
                report_dependencies = map(lambda x: x[1], sorted(report_dependencies.items(), key=lambda x: x[0]))

                generator = hydra.utils.instantiate(report_cfg)
                generator.name = report_name
                generator_param_names = list(inspect.signature(generator.generate).parameters)

                if num_deps > 0:
                    for dep_name in report_dependencies:
                        generator_kwargs = dict()

                        dep_outputs = generated[dep_name]
                        for dep_output in dep_outputs:
                            generator_kwargs = {
                                **generator_kwargs,
                                **{
                                    param_name: dep_output[param_name]
                                    for param_name in generator_param_names
                                    if param_name in dep_output
                                }
                            }

                            output = generator.generate(**generator_kwargs)
                            if output is not None:
                                generated[report_name].extend(output)
                else:
                    output = generator.generate()
                    if output is not None:
                        generated[report_name].extend(output)

                self.log(logging.INFO, f'"{report_name}" was generated successfully.')
            except:  # noqa
                self.log(logging.ERROR, f'"{report_name}" failed. Details:\n' + traceback.format_exc())

    def _init_data(
        self,
        cfg: DictConfig
    ) -> persistence.Data:

        data = None

        self._dataset = hydra.utils.instantiate(cfg.experiment.dataset)
        self._splitter = hydra.utils.instantiate(cfg.experiment.split)

        if self._dataset is not None:
            self._samples, self._targets, *self._other = self._dataset.load()

            if self._samples.isnull().sum().sum() > 0:
                self.log(logging.WARNING, 'Data contain NaN values')

            data_md5 = self._dataset.md5
            data = self.db_connection.query(persistence.Data).filter(persistence.Data.md5 == data_md5).first()
            if data is None:
                data = persistence.Data(
                    name=self._dataset.name,
                    meta=self._dataset.meta,
                    num_samples=self._dataset.num_samples,
                    num_features=self._dataset.num_features,
                    md5=data_md5
                )

        return data

    @staticmethod
    def _init_pool(
        samples: pd.DataFrame,
        targets: pd.DataFrame,
        train_ids: List[np.ndarray],
        test_ids: List[np.ndarray]
    ) -> None:

        global SAMPLES, TARGETS, TRAIN_IDS, TEST_IDS

        SAMPLES, TARGETS = samples, targets
        TRAIN_IDS, TEST_IDS = train_ids, test_ids

    @staticmethod
    def _run_task(
        idx: int,
        task: DictConfig,
        fold: int
    ) -> Tuple[
        int,
        Optional[int],
        bool,
        Union[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], str]
    ]:

        success = False
        out = None

        try:
            train_ids, test_ids = TRAIN_IDS[fold], TEST_IDS[fold]

            X_train = SAMPLES.iloc[train_ids]  # noqa
            y_train = TARGETS.iloc[train_ids]

            X_test = SAMPLES.iloc[test_ids]  # noqa

            for name in task.transformations.keys():
                transform = hydra.utils.instantiate(task.transformations[name])

                transform.fit(X_train)

                X_train = transform.transform(X_train)  # noqa
                X_test = transform.transform(X_test)  # noqa

            model_name = next(iter(task.model.keys()))
            model = hydra.utils.instantiate(task.model[model_name])

            model.fit(X_train, y_train)

            out = model.predict(X_train), model.predict(X_test), RegulAS._get_feature_scores(model)
            success = True
        except:  # noqa
            out = traceback.format_exc()
        finally:
            return idx, fold, success, out

    def _process_task_result(
        self,
        result: Tuple[
            int,
            Optional[int],
            bool,
            Union[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], str]
        ]
    ) -> None:

        pipeline = None

        try:
            idx, fold, success, out = result

            pipeline = persistence.Pipeline(experiment=self._experiment, success=success, fold=fold)

            self._num_finished += 1
            task_desc = f'Task #{idx} ({self._num_finished}/{self._num_tasks})'

            _, task, *_ = self._tasks[idx]

            for position, (name, transformation_cfg) in enumerate(task.transformations.items(), 1):
                self._assign_transformation(pipeline, transformation_cfg, name, position)

            model_name = next(iter(task.model.keys()))
            model_cfg = task.model[model_name]
            self._assign_transformation(pipeline, model_cfg, model_name, type_=persistence.Transformation.Type.MODEL)

            if success:
                y_pred_train, y_pred_test, feature_scores = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], out)
                run_idx = 0 if OmegaConf.is_missing(HydraConfig.get().job, 'num') else HydraConfig.get().job.num
                self.log(logging.INFO, f'[Run #{run_idx}] {task_desc} finished successfully.')

                for idx_pred, idx_true in enumerate(self._train_ids[fold]):
                    persistence.Prediction(
                        pipeline=pipeline,
                        sample_name=self._samples.index[idx_true],
                        true_value=dump_ndarray(self._targets.iloc[idx_true].values.astype(np.float16)),
                        predicted_value=dump_ndarray(y_pred_train[idx_pred].astype(np.float16)),
                        training=True
                    )

                for idx_pred, idx_true in enumerate(self._test_ids[fold]):
                    persistence.Prediction(
                        pipeline=pipeline,
                        sample_name=self._samples.index[idx_true],
                        true_value=dump_ndarray(self._targets.iloc[idx_true].values.astype(np.float16)),
                        predicted_value=dump_ndarray(y_pred_test[idx_pred].astype(np.float16)),
                        training=False
                    )

                if feature_scores is not None and len(feature_scores) == len(self._samples.columns):
                    for feature_idx, feature_name in enumerate(self._samples.columns):
                        persistence.FeatureRanking(
                            pipeline=pipeline,
                            feature=feature_name,
                            score=float(feature_scores[feature_idx])
                        )
            else:
                tb = out
                self.log(logging.ERROR, f'{task_desc} failed. Details:\n' + tb)
        except:  # noqa
            self.log(logging.ERROR, f'Could not process result. Details:\n{traceback.format_exc()}')
        finally:
            if pipeline is not None:
                self.db_connection.add(pipeline)
                self.db_connection.commit()

    def _get_dao(
        self,
        dao_cls: Type[persistence.RegulASTable],
        condition: Optional[Any] = None,
        **kwargs
    ) -> persistence.RegulASTable:

        dao: Union[Query, persistence.RegulASTable] = self.db_connection.query(dao_cls)

        if condition is not None:
            dao = dao.filter(condition)

        dao = cast(persistence.RegulASTable, dao.first())

        if dao is None:
            dao = dao_cls(**kwargs)

        return dao

    def _assign_transformation(
        self,
        pipeline: persistence.Pipeline,
        cfg: DictConfig,
        alias: Optional[str] = None,
        position: int = 1,
        type_: persistence.Transformation.Type = persistence.Transformation.Type.TRANSFORM
    ) -> persistence.Pipeline:

        fqn = cfg['_target_']

        if alias is None:
            alias = fqn.rsplit('.', 1)[-1]

        transformation_src = inspect.getsource(hydra.utils.get_class(fqn))

        base_package, *_ = fqn.split('.')
        module = sys.modules.get(base_package, None)

        version = None
        if module is not None:
            version = getattr(module, '__version__', None)

        if version is None:
            src_md5 = hashlib.md5()
            src_md5.update(transformation_src.encode())
            version = src_md5.hexdigest()

        transformation = self._get_dao(
            persistence.Transformation,
            and_(
                persistence.Transformation.fqn == fqn,
                persistence.Transformation.version == version,
                persistence.Transformation.type_ == type_
            ),
            fqn=fqn,
            version=version,
            source=transformation_src,
            type_=type_
        )

        transformation_sequence = persistence.TransformationSequence(
            pipeline=pipeline,
            transformation=transformation,
            alias=alias,
            position=position
        )
        self._assign_hyper_parameters(transformation_sequence, cfg)

        pipeline.transformations.append(transformation_sequence)

        return pipeline

    def _assign_hyper_parameters(
        self,
        transformation_sequence: persistence.TransformationSequence,
        cfg: DictConfig
    ) -> persistence.TransformationSequence:

        model_hyper_parameters = list()
        for name, value in cfg.items():
            if name == '_target_':
                continue

            model_hyper_parameters.append((name, str(value)))

            hyper_parameter = self._get_dao(
                persistence.HyperParameter,
                condition=(persistence.HyperParameter.name == name),
                name=name
            )

            hyper_parameter_value = persistence.HyperParameterValue(
                transformation=transformation_sequence,
                hyper_parameter=hyper_parameter,
                value=str(value)
            )
            transformation_sequence.hyper_parameters.append(hyper_parameter_value)

        model_hyper_parameters = ', '.join([
            f'{name}: {value}' for name, value in sorted(model_hyper_parameters, key=lambda x: x[0])
        ])
        model_hyper_parameters = f'{{{model_hyper_parameters}}}'

        hyper_parameters_md5 = hashlib.md5()
        hyper_parameters_md5.update(model_hyper_parameters.encode())
        transformation_sequence.md5 = hyper_parameters_md5.hexdigest()

        return transformation_sequence

    @staticmethod
    def _get_feature_scores(model) -> np.ndarray:
        scores = None

        for name in RegulAS.SCORE_ATTR_NAMES:
            scores = getattr(model, name, None)

            if scores is not None:
                scores = scores.flatten()
                break

        return scores
