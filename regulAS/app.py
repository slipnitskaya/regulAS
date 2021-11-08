import os
import copy
import logging

import hydra  # noqa

from itertools import product
from collections import defaultdict

from sklearnex import patch_sklearn

from omegaconf import OmegaConf, DictConfig, ListConfig

from typing import Any, Dict, List, Union, Callable, Optional

from .core import RegulAS


TraversalAction = Callable[..., DictConfig]


patch_sklearn()


def traverse_config(root: Union[DictConfig, ListConfig],
                    path: Optional[List[str]] = None,
                    action: TraversalAction = lambda x: x,
                    action_kwargs: Optional[Dict[str, Any]] = None) -> DictConfig:

    if path is None:
        path = list()

    if action_kwargs is None:
        action_kwargs = dict()

    root = action(root, path, **action_kwargs)

    for name in root.keys():
        node = root[name]
        if isinstance(node, (DictConfig, ListConfig)):
            root[name] = traverse_config(node, path + [name], action, action_kwargs)

    return root


def clone_task_tree(node: DictConfig, path: List[str], trees_: List) -> DictConfig:
    fqn = '.'.join(path)

    if '_varargs_' not in fqn and path and '_varargs_' not in path[-1]:
        for tree_ in trees_:
            if OmegaConf.select(tree_, fqn) is None:
                OmegaConf.update(tree_, fqn, OmegaConf.create(dict()))

            if path:
                for name_ in list(node.keys()):
                    value = node[name_]
                    if not OmegaConf.is_config(value):
                        OmegaConf.update(tree_, f'{fqn}.{name_}', value)

    return node


def expand_varargs(node: DictConfig, path: List[str], trees_: List) -> DictConfig:
    fqn = '.'.join(path)

    if '_varargs_' in node:
        fqn_mapping = dict()
        fqns = defaultdict(list)

        for name_ in node['_varargs_'].keys():
            for val_name in node['_varargs_'][name_].keys():
                fqn_src = f'_varargs_.{name_}'
                fqn_dst = f'{fqn}.{name_}'

                fqn_mapping[fqn_src] = fqn_dst
                fqns[fqn_src].append(f'{fqn_src}.{val_name}')

        cloned_trees = list()

        for varargs in product(*[fqns_ for fqns_ in fqns.values()]):
            for tree_ in trees_:
                cloned = copy.deepcopy(tree_)

                for fqn_src in varargs:
                    fqn_dst = fqn_mapping[fqn_src.rsplit('.', 1)[0]]

                    value = OmegaConf.select(node, fqn_src)
                    OmegaConf.update(cloned, fqn_dst, value)

                cloned_trees.append(cloned)

        trees_.clear()
        trees_.extend(cloned_trees)

        del node['_varargs_']

    return node


def prepare_tasks(cfg: DictConfig) -> List[DictConfig]:
    task_idx: int = 0
    cfg.tasks = OmegaConf.create(dict())
    for _, pipeline in cfg.experiment.pipelines.items():
        task_trees = [OmegaConf.create(dict())]

        traverse_config(pipeline, action=clone_task_tree, action_kwargs={'trees_': task_trees})
        traverse_config(pipeline, action=expand_varargs, action_kwargs={'trees_': task_trees})

        for task_tree in task_trees:
            cfg.tasks[str(task_idx)] = task_tree
            task_idx += 1

    tasks: List[DictConfig] = [task for idx, task in cfg.tasks.items()]

    return tasks


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), 'conf'), config_name='default')
def run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    cfg.experiment = OmegaConf.load(
        os.path.join(hydra.utils.get_original_cwd(), f'{cfg.experiment}.yaml')
    )

    override_names = set(cfg.keys()) & set(cfg.experiment.keys())
    if override_names:
        for name in override_names:
            OmegaConf.update(cfg, 'experiment', OmegaConf.masked_copy(cfg, name), merge=True)
            del cfg[name]

    OmegaConf.resolve(cfg)
    cfg = traverse_config(
        cfg,
        action=lambda n, _: OmegaConf.create({
            str(idx): child for idx, child in enumerate(n)
        }) if OmegaConf.is_list(n) else n
    )

    init_status = RegulAS().init(cfg)

    if init_status & RegulAS.InitStatus.FAILED:
        RegulAS.log(logging.WARNING, 'Stopping.')
        return

    if init_status & RegulAS.InitStatus.ALLOW_SUBMIT:
        RegulAS.log(logging.INFO, 'Preparing tasks to submit...')
        tasks = prepare_tasks(cfg)
        RegulAS().submit(tasks)

    if init_status & RegulAS.InitStatus.ALLOW_REPORTS:
        RegulAS.log(logging.INFO, 'Generating reports...')
        RegulAS().generate(cfg.experiment.reports)


if __name__ == '__main__':
    run()
