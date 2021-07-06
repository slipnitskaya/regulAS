import os
import sys
import enum
import hashlib
import inspect
import logging
import traceback
import multiprocessing as mp

import hydra
import numpy as np
import pandas as pd

from typing import cast, Any, Dict, List, Type, Tuple, Union, Optional, Collection

from omegaconf import DictConfig, OmegaConf

from sqlalchemy import and_, create_engine

from sqlalchemy.orm import Session, Query
from sqlalchemy.engine.url import URL, make_url

from regulAS import persistence
from regulAS.utils.base import Loader, Split


SAMPLES: Optional[pd.DataFrame] = None
TARGETS: Optional[pd.DataFrame] = None


class Singleton(type):

    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class RegulAS(metaclass=Singleton):

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

    class Status(enum.Enum):
        SUCCESS = enum.auto()
        FAILED = enum.auto()

    def __init__(self):
        super(RegulAS, self).__init__()

    def init(self, cfg: DictConfig):
        logging.captureWarnings(True)

        self._num_processes = cfg.num_processes or mp.cpu_count()

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
        self._session = Session(bind=create_engine(db_url))

        persistence.create_schema(db_url, self._session.bind)

        data = self._init_data(cfg)

        cfg_str = OmegaConf.to_yaml(cfg)
        cfg_str_md5 = hashlib.md5()
        cfg_str_md5.update(cfg_str.encode())
        cfg_str_md5.update(cfg.experiment.name.encode())
        cfg_str_md5.update(data.md5.encode())
        experiment_md5 = cfg_str_md5.hexdigest()

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
            return self.Status.FAILED

        self._experiment = persistence.Experiment(
            name=cfg.experiment.name,
            data=data,
            config=cfg_str,
            md5=experiment_md5,
            random_seed=cfg.random_state
        )
        self.db_connection.add(self._experiment)
        self.db_connection.commit()

        self._train_ids, self._test_ids = list(), list()
        for train_ids, test_ids in self._splitter.split(self._samples, self._targets, *self._other):
            self._train_ids.append(train_ids)
            self._test_ids.append(test_ids)

        self._tasks = dict()

        return self.Status.SUCCESS

    def _init_data(self, cfg: DictConfig) -> persistence.Data:
        self._dataset = hydra.utils.instantiate(cfg.experiment.dataset)
        self._splitter = hydra.utils.instantiate(cfg.experiment.split)

        self._samples, self._targets, *self._other = self._dataset.load()

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
    def _pool_init(samples: pd.DataFrame, targets: pd.DataFrame):
        global SAMPLES, TARGETS
        SAMPLES, TARGETS = samples, targets

    @staticmethod
    def _runner(idx: int,
                task: DictConfig,
                train_ids: np.ndarray,
                test_ids: np.ndarray) -> Tuple[int, bool, Union[np.ndarray, str]]:

        success = False
        out = None

        try:
            X_train = SAMPLES.iloc[train_ids]
            y_train = TARGETS.iloc[train_ids]

            X_test = SAMPLES.iloc[test_ids]

            for name in task.transformations.keys():
                transform = hydra.utils.instantiate(task.transformations[name])

                transform.fit(X_train)

                X_train = transform.transform(X_train)
                X_test = transform.transform(X_test)

            model_name = next(iter(task.model.keys()))
            model = hydra.utils.instantiate(task.model[model_name])

            model.fit(X_train, y_train)

            out = model.predict(X_test)
            success = True
        except:
            out = traceback.format_exc()
        finally:
            return idx, success, out

    def _process_result(self, result: Tuple[int, bool, Union[np.ndarray, str]]):
        idx, success, out = result

        self._num_finished += 1
        task_desc = f'Task #{idx} ({self._num_finished}/{self._num_tasks})'

        _, task, *_ = self._tasks[idx]

        pipeline = persistence.Pipeline(
            experiment=self._experiment,
            success=success
        )

        for position, (name, transformation_cfg) in enumerate(task.transformations.items(), 1):
            transformation_src = inspect.getsource(hydra.utils.get_class(transformation_cfg['_target_']))

            base_package, *_ = transformation_cfg['_target_'].split('.')
            transformation_module = sys.modules.get(base_package, None)

            transformation_version = None
            if transformation_module is not None:
                transformation_version = getattr(transformation_module, '__version__', None)

            if transformation_version is None:
                src_md5 = hashlib.md5()
                src_md5.update(transformation_src.encode())
                transformation_version = src_md5.hexdigest()

            transformation = self._get_dao(
                persistence.Transformation,
                condition=and_(
                    persistence.Transformation.fqn == transformation_cfg['_target_'],
                    persistence.Transformation.version == transformation_version,
                    persistence.Transformation.type_ == persistence.Transformation.Type.TRANSFORM
                ),
                fqn=transformation_cfg['_target_'],
                version=transformation_version,
                source=transformation_src,
                type_=persistence.Transformation.Type.TRANSFORM
            )

            transformation_sequence = persistence.TransformationSequence(
                pipeline=pipeline,
                transformation=transformation,
                position=position
            )

            for hp_name, value in transformation_cfg.items():
                if hp_name == '_target_':
                    continue

                hyper_parameter = self._get_dao(
                    persistence.HyperParameter,
                    condition=(persistence.HyperParameter.name == hp_name),
                    name=hp_name
                )

                hyper_parameter_value = persistence.HyperParameterValue(
                    transformation=transformation_sequence,
                    hyper_parameter=hyper_parameter,
                    value=str(value)
                )
                transformation_sequence.hyper_parameters.append(hyper_parameter_value)

            pipeline.transformations.append(transformation_sequence)

        model_name = next(iter(task.model.keys()))
        model_cfg = task.model[model_name]

        model_src = inspect.getsource(hydra.utils.get_class(model_cfg['_target_']))

        base_package, *_ = model_cfg['_target_'].split('.')
        model_module = sys.modules.get(base_package, None)

        model_version = None
        if model_module is not None:
            model_version = getattr(model_module, '__version__', None)

        if model_version is None:
            src_md5 = hashlib.md5()
            src_md5.update(model_src.encode())
            model_version = src_md5.hexdigest()

        model = self._get_dao(
            persistence.Transformation,
            and_(
                persistence.Transformation.fqn == model_cfg['_target_'],
                persistence.Transformation.version == model_version,
                persistence.Transformation.type_ == persistence.Transformation.Type.MODEL
            ),
            fqn=model_cfg['_target_'],
            version=model_version,
            source=model_src,
            type_=persistence.Transformation.Type.MODEL
        )

        transformation_sequence = persistence.TransformationSequence(
            pipeline=pipeline,
            transformation=model
        )

        for hp_name, value in model_cfg.items():
            if hp_name == '_target_':
                continue

            hyper_parameter = self._get_dao(
                persistence.HyperParameter,
                condition=(persistence.HyperParameter.name == hp_name),
                name=hp_name
            )

            hyper_parameter_value = persistence.HyperParameterValue(
                transformation=transformation_sequence,
                hyper_parameter=hyper_parameter,
                value=str(value)
            )
            transformation_sequence.hyper_parameters.append(hyper_parameter_value)

        pipeline.transformations.append(transformation_sequence)

        self.db_connection.add(pipeline)
        self.db_connection.commit()

        if success:
            y_pred = out
            self.log(logging.INFO, f'{task_desc} finished successfully.')

            # TODO: write into DB
        else:
            tb = out
            self.log(logging.ERROR, f'{task_desc} failed. Details:\n' + tb)

    def submit(self, tasks: Collection[DictConfig]) -> None:
        self.log(
            logging.INFO,
            f'{len(tasks)} tasks were prepared for '
            f'{self._splitter.n_splits}-fold cross-validation '
            f'using {self._num_processes} workers '
            f'({len(tasks) * self._splitter.n_splits} tasks overall)'
        )

        task_idx = 1
        for task in tasks:
            for fold in range(self._splitter.n_splits):
                self._tasks[task_idx] = task_idx, task, self._train_ids[fold], self._test_ids[fold]
                task_idx += 1

        with mp.Pool(
            processes=self._num_processes,
            initializer=self._pool_init,
            initargs=(self._samples, self._targets)
        ) as pool:

            self._num_tasks = len(self._tasks)
            self._num_finished = 0

            for idx, task in self._tasks.items():
                pool.apply_async(func=self._runner, args=task, callback=self._process_result)
                self.log(logging.INFO, f'Task {idx}/{self._num_tasks} has been submitted.')

            pool.close()
            pool.join()

            self.db_connection.commit()

    def _get_dao(self,
                 dao_cls: Type[persistence.RegulASTable],
                 condition: Optional[Any] = None,
                 **kwargs) -> persistence.RegulASTable:

        dao: Union[Query, persistence.RegulASTable] = self.db_connection.query(dao_cls)

        if condition is not None:
            dao = dao.filter(condition)

        dao = cast(persistence.RegulASTable, dao.first())

        if dao is None:
            dao = dao_cls(**kwargs)

        return dao

    @property
    def db_connection(self):
        return self._session

    @staticmethod
    def log(level, msg, *args, **kwargs):
        RegulAS._logger.log(level, msg, *args, **kwargs)
