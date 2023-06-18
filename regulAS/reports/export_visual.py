import os
import re
import difflib

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

import hydra  # noqa

import regulAS.persistence as persistence

from sqlalchemy import and_, not_
from matplotlib.figure import Axes, Figure

from regulAS import RegulAS
from regulAS.reports import Report
from regulAS.utils import load_ndarray

from typing import cast, Dict, Tuple, Optional


class ModelPerformanceBarGraphReport(Report):

    fonttitle: int = 18
    fontsize: int = 14

    def __init__(
        self,
        output_dir: str,
        bar_targets: Dict[str, str],
        bar_errors: Optional[Dict[str, str]] = None,
        greater_is_better: bool = False,
        fig_width: float = 8.0,
        fig_height: float = 8.0,
        dpi: Optional[float] = None,
        title: str = 'PSI prediction error of tested models',
        x_label: str = 'models',
        y_label: str = 'objective',
        y_lim_bottom: float = 0.0,
        y_lim_top: Optional[float] = None
    ):
        super(ModelPerformanceBarGraphReport, self).__init__()

        self.output_dir = output_dir
        self.bar_targets = bar_targets
        self.bar_errors = bar_errors
        self.greater_is_better = greater_is_better
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.y_lim_bottom = y_lim_bottom
        self.y_lim_top = y_lim_top

    @property
    def figsize(self):
        if self.fig_width is None and self.fig_height is None:
            return None

        if self.fig_width is not None and self.fig_height is None:
            self.fig_height = self.fig_width

        if self.fig_width is None and self.fig_height is not None:
            self.fig_width = self.fig_height

        return self.fig_width, self.fig_height

    @property
    def y_lim(self):
        return self.y_lim_bottom, self.y_lim_top

    def generate(self, df: pd.DataFrame) -> None:
        bar_targets = [
            difflib.get_close_matches(target, df.columns, cutoff=0.25, n=1).pop(0)
            for _, target in sorted(self.bar_targets.items(), key=lambda x: x[0])
        ]
        pretty_targets = list(map(
            lambda col, m=lambda c: re.match('.*:(train|test):.*', c): (
                lambda m_=m(col): col if m_ is None else f'{m_.group(1)} data'
            )(),
            bar_targets
        ))

        if self.bar_errors is not None:
            bar_errors = [
                difflib.get_close_matches(target, df.columns, n=1).pop(0)
                for _, target in sorted(self.bar_errors.items(), key=lambda x: x[0])
            ]
        else:
            bar_errors = [None] * len(bar_targets)

        perf_groups = df.droplevel(['data', 'data_md5', 'experiment_name', 'experiment_md5']).groupby(level='model')

        top_models = pd.concat([
            model_df.iloc[[0]].drop(columns='hyper_parameters') for _, model_df in perf_groups
        ]).sort_values(
            by=bar_targets[:1],
            ascending=not self.greater_is_better
        ).rename(
            columns=dict(zip(bar_targets, pretty_targets))
        )

        top_models = top_models.set_index(
            top_models.index.to_flat_index()
        ).rename(index=df.attrs['model_aliases'])

        fig, ax = cast(Tuple[Figure, Axes], plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi))

        top_models[pretty_targets].plot(
            ax=ax,
            kind='bar',
            yerr=top_models[bar_errors].rename(
                columns=dict(zip(bar_errors, pretty_targets))
            ),
            rot=45
        )
        ax.set_title(self.title, fontsize=self.fonttitle)
        ax.set_xlabel(self.x_label, fontsize=self.fontsize)
        ax.set_ylabel(self.y_label, fontsize=self.fontsize)
        ax.set_ylim(*self.y_lim)

        plt.grid(axis='y')

        data_name, data_md5, *_ = map(lambda x: x.pop(), (set(item) for item in zip(*df.index)))
        df_title = '-'.join([df.attrs.get('title', ''), self.name, data_name, data_md5])

        if os.path.isabs(self.output_dir):
            output_dir = self.output_dir
        else:
            output_dir = os.path.join(hydra.utils.get_original_cwd(), self.output_dir)

        path_to_output = os.path.abspath(os.path.join(output_dir, f'{df_title}.png'))
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

        plt.savefig(path_to_output, bbox_inches='tight')
        plt.close(fig)


class ModelPredictionsScatterPlotReport(Report):

    fonttitle: int = 18
    fontsize: int = 14

    def __init__(
        self,
        output_dir: str,
        correlation_method: str = 'spearmanr',
        fig_width: float = 8.0,
        fig_height: float = 8.0,
        dpi: Optional[float] = None,
        title: str = 'Modelling of PSI',
        x_label: str = 'PSI observed',
        y_label: str = 'PSI modelled'
    ):
        super(ModelPredictionsScatterPlotReport, self).__init__()

        self.output_dir = output_dir
        self.correlation_method = correlation_method
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    @staticmethod
    def _fit_line_to_points(
        x: np.ndarray,
        y: np.ndarray,
        val_min: float = 0.0,
        val_max: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:

        poly_coeffs = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(poly_coeffs)

        out_x = np.array([val_min, val_max])
        out_y = fit_fn(out_x)

        return out_x, out_y

    @staticmethod
    def _print_correlation(
        x: np.array,
        y: np.array,
        method: str = 'spearmanr',
        alpha: float = 0.005
    ) -> str:

        if method == 'pearsonr':
            pcc, p_value = scipy.stats.pearsonr(x, y)
        elif method == 'spearmanr':
            pcc, p_value = scipy.stats.spearmanr(x, y, nan_policy='omit')
        else:
            raise ValueError(f'Unknown correlation method: {method}')

        return r'$\rho$={:.2f} ($P${}{})'.format(pcc, "<" if p_value < alpha else ">", alpha)

    @property
    def figsize(self):
        if self.fig_width is None and self.fig_height is None:
            return None

        if self.fig_width is not None and self.fig_height is None:
            self.fig_height = self.fig_width

        if self.fig_width is None and self.fig_height is not None:
            self.fig_width = self.fig_height

    def generate(self, df: pd.DataFrame) -> None:
        conn = RegulAS().db_connection

        (
            data_name,
            data_md5,
            experiment_name,
            experiment_md5,
            *_
        ) = map(lambda x: x.pop(), (set(item) for item in zip(*df.index)))

        perf_groups = df.droplevel(['data', 'data_md5', 'experiment_name', 'experiment_md5']).groupby(level='model')

        top_models = pd.concat([
            model_df.iloc[[0]].drop(columns='hyper_parameters') for _, model_df in perf_groups
        ])
        top_models = top_models.set_index(
            top_models.index.to_flat_index()
        )

        for (model_fqn, hyper_parameters_md5), model_scores in top_models.iterrows():
            model_alias = df.attrs['model_aliases'][(model_fqn, hyper_parameters_md5)]

            predictions = conn.query(
                persistence.Prediction
            ).join(
                persistence.Pipeline,
                persistence.Experiment,
                persistence.Data,
                persistence.TransformationSequence,
                persistence.Transformation
            ).filter(
                and_(
                    persistence.Data.md5 == data_md5,
                    persistence.Experiment.md5 == experiment_md5,
                    not_(persistence.Prediction.training),
                    persistence.Transformation.type_ == persistence.Transformation.Type.MODEL,
                    persistence.Transformation.fqn == model_fqn,
                    persistence.TransformationSequence.md5 == hyper_parameters_md5
                )
            )

            y_true, y_pred = map(
                lambda x: np.array(x).reshape(len(x), -1).squeeze().astype(np.float32),
                zip(*map(
                    lambda entry: (load_ndarray(entry.true_value), load_ndarray(entry.predicted_value)),
                    predictions
                ))
            )

            fig = plt.figure(figsize=self.figsize)

            plt.scatter(
                x=y_true,
                y=np.clip(y_pred, a_min=0.0, a_max=1.0),
                s=10
            )
            plt.title(f'{self.title}\n({model_alias})', fontsize=self.fonttitle, y=1.01)
            plt.annotate(
                f'{self._print_correlation(y_true, y_pred, method=self.correlation_method)}',
                size=self.fontsize,
                xy=(0.2, 0.02)
            )

            plt.xlabel(self.x_label, fontsize=self.fontsize)
            plt.ylabel(self.y_label, fontsize=self.fontsize)

            plt.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                c='gray',
                alpha=0.7,
                ls='--',
                label='y=x line'
            )

            # plot fitting line
            bnd_low = np.min([np.min(y_true), np.min(y_pred)])
            bnd_high = np.max([np.max(y_true), np.max(y_pred)])
            fitted = self._fit_line_to_points(y_true, y_pred, val_min=bnd_low, val_max=bnd_high)
            plt.plot(*fitted, c='r', linewidth=1.0, label='Linear fitting')
            plt.legend(loc='upper left')
            plt.xlim(left=0.0, right=1.0)
            plt.ylim(bottom=0.0, top=1.0)

            df_title = '-'.join([
                df.attrs.get('title', ''),
                self.name,
                self.correlation_method,
                model_alias,
                hyper_parameters_md5,
                data_name,
                data_md5
            ])

            if os.path.isabs(self.output_dir):
                output_dir = self.output_dir
            else:
                output_dir = os.path.join(hydra.utils.get_original_cwd(), self.output_dir)

            path_to_output = os.path.abspath(os.path.join(output_dir, f'{df_title}.png'))
            os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

            plt.savefig(fname=path_to_output, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)


class FeatureRankingBarGraphReport(Report):

    fonttitle: int = 18
    fontsize: int = 14
    labelsize: int = None

    def __init__(
        self,
        output_dir: str,
        top_k: int = 10,
        fig_width: float = 8.0,
        fig_height: float = 8.0,
        dpi: Optional[float] = None,
        title: str = 'Most{top_k} relevant regulators\n({model_alias})',
        x_label: str = 'relevance score',
        y_label: str = 'regulators'
    ):
        super(FeatureRankingBarGraphReport, self).__init__()

        self.output_dir = output_dir
        self.top_k = top_k
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    @property
    def figsize(self):
        if self.fig_width is None and self.fig_height is None:
            return None

        if self.fig_width is not None and self.fig_height is None:
            self.fig_height = self.fig_width

        if self.fig_width is None and self.fig_height is not None:
            self.fig_width = self.fig_height

        return self.fig_width, self.fig_height

    def generate(self, df: pd.DataFrame) -> None:
        if isinstance(df.index, pd.MultiIndex):
            index = df.index
            axis = 0
        else:
            index = df.columns
            axis = 1

        (
            data_name,
            data_md5,
            experiment_name,
            experiment_md5,
            *_
        ) = map(lambda x: x.pop(), (set(item) for item in zip(*index)))

        score_groups = df.droplevel(
            level=['data', 'data_md5', 'experiment_name', 'experiment_md5'], axis=axis
        ).groupby(
            level=['model', 'hyper_parameters_md5'], axis=axis
        )

        for (model_fqn, hyper_parameters_md5), feature_scores_df in score_groups:
            model_alias = df.attrs['model_aliases'][(model_fqn, hyper_parameters_md5)]

            feature_scores_df = feature_scores_df.droplevel(
                level=['model', 'hyper_parameters_md5'], axis=axis
            ).sort_values(
                by=['score:mean', 'score:std'],
                key=abs,
                ascending=False
            )

            if self.top_k is not None:
                feature_scores_df = feature_scores_df.iloc[:self.top_k]

            errors = np.stack(
                (
                    np.where(
                        (feature_scores_df['score:mean'].abs() - feature_scores_df['score:std']) > 0.0,
                        feature_scores_df['score:std'],
                        feature_scores_df['score:mean'].abs()
                    ),
                    feature_scores_df['score:std']
                ),
                axis=1
            ).T
            colors = ['C3' if value > 0.0 else 'C0' for value in feature_scores_df['score:mean'].values]

            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

            feature_scores_df['score:mean'].abs().plot(
                ax=ax,
                kind='barh',
                legend=False,
                xerr=errors,
                color=colors
            )

            title_format_kwargs = dict()
            if '{model_alias}' in self.title:
                title_format_kwargs['model_alias'] = model_alias
            if '{top_k}' in self.title and self.top_k is not None:
                title_format_kwargs['top_k'] = f'-{self.top_k}'

            plt.title(
                self.title.format(**title_format_kwargs),
                fontsize=self.fonttitle,
                y=1.01
            )
            ax.set_xlabel(self.x_label, fontsize=self.fontsize)
            ax.set_ylabel(self.y_label, fontsize=self.fontsize)
            if self.labelsize is not None:
                ax.tick_params(labelsize=self.labelsize)

            ax.set_yticklabels([
                f'{feature}\n({score:.3e})'
                for feature, score
                in zip(feature_scores_df.index, feature_scores_df['score:mean'])
            ])

            plt.gca().invert_yaxis()
            plt.grid(axis='x')

            df_title = '-'.join([
                df.attrs.get('title', ''),
                self.name,
                model_alias,
                hyper_parameters_md5,
                data_name,
                data_md5
            ])

            if os.path.isabs(self.output_dir):
                output_dir = self.output_dir
            else:
                output_dir = os.path.join(hydra.utils.get_original_cwd(), self.output_dir)

            path_to_output = os.path.abspath(os.path.join(output_dir, f'{df_title}.png'))
            os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

            plt.savefig(fname=path_to_output, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
