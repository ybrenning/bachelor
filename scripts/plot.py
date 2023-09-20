import json
import logging
import mlflow
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mlflow import get_experiment_by_name, search_runs
from pathlib import Path

plt.rcParams['axes.labelsize'] = '22'
plt.rcParams['font.size'] = '10'
#plt.rcParams['text.textsize'] = '10'
plt.rcParams['xtick.labelsize'] = '16'
plt.rcParams['ytick.labelsize'] = '16'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = '1.5'

plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1


from active_learning_lab.experiments.active_learning.postprocess.learning_curves import plot_learning_curve
from active_learning_lab.utils.results import get_results, check_for_duplicates, check_for_reproduciblity, assemble_df


logger = logging.getLogger(__name__)


def get_results(exp_name: str):
    experiment = get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id

    runs = search_runs(experiment_ids=[experiment_id])
    return runs[(~runs.run_id.isna()) & (runs.status == 'FINISHED')]


def check_for_duplicates(df, columns=['params.dataset_name', 'params.classifier_name', 'params.query_strategy']):
    df_tmp = df.groupby(columns).size().reset_index()

    duplicates = df_tmp[df_tmp[0] > 1]

    if len(duplicates) > 0:
        duplicates = df.set_index(columns).loc[duplicates.itertuples(index=False, name=None)]

        duplicates = duplicates.groupby(columns).agg({'run_id': list}).reset_index()
        duplicates = duplicates.rename(columns={'run_id': 'run_ids'})

        # TODO: columns must refer to string columns
        error_cases = ['\t' + '_'.join([row[col] for col in columns]) + ': ' + str(row['run_ids'])
                       for _, row in duplicates.iterrows()]
        error_cases = '\n'.join(error_cases)

        raise ValueError('Multiple results were found for the following strategies:\n'
                         f'{error_cases}')

    return df


# TODO: this can be improved by extending assemble_dfs to include all parameters
def check_for_reproduciblity(df):

    for row in df.itertuples():
        registry_uri = re.sub('^file://', '', mlflow.get_registry_uri())
        registry_uri += f'/{row.experiment_id}/{row.run_id}'

        config_path = Path(registry_uri).joinpath('artifacts/config.json')

        try:
            config = json.load(config_path.open())
            print(config['general']['max_reproducibility'])
            if not config['general']['max_reproducibility']:
                logger.warning(f'{row.run_id} ({row.params.classifier_name} / {row.params.dataset_name} / '
                               f'{row.params.query_strategy}) has not set the max_reproducibility flag')
        except FileNotFoundError:
            # raised when experiment still running or aborted
            logger.warning(f'Run {row.run_id} has no params/config (still running or aborted)')

    return df


def assemble_df(df, csv_name, index_col=None):
    dfs = []

    for row in df.itertuples():
        registry_uri = re.sub('^file://', '', mlflow.get_registry_uri())
        registry_uri += f'/{row.experiment_id}/{row.run_id}'

        try:
            csv_path = Path(registry_uri).joinpath(f'artifacts/{csv_name}')

            df_tmp = pd.read_csv(csv_path, header=0, index_col=index_col)

            # +1 because row._fields has an additional field "Index" at the beginning
            df_tmp['dataset_name'] = row[df.columns.get_loc('params.dataset_name')+1]
            df_tmp['classifier'] = row[df.columns.get_loc('params.classifier_name')+1]
            df_tmp['query_strategy'] = row[df.columns.get_loc('params.query_strategy')+1]

            dfs.append(df_tmp)
        except FileNotFoundError:
            # raised when experiment still running or aborted
            logger.warning(f'folder {row.run_id} has no params/config (still running or aborted)')

    df_all = pd.concat(dfs)
    return df_all


def plot_dataset(results, dataset, ax):
    results_random = results.query(f'`params.dataset_name` == "{dataset}" and `params.query_strategy` == "random"')
    results_bt = results.query(f'`params.dataset_name` == "{dataset}" and `params.query_strategy` == "lc-bt"')
    results_gc = results.query(f'`params.dataset_name` == "{dataset}" and `params.query_strategy` == "gc"')

    df_acc_random = assemble_df(results_random, 'results.csv')
    df_acc_bt = assemble_df(results_bt, 'results.csv')
    df_acc_gc = assemble_df(results_gc, 'results.csv')

    sample_sizes = df_acc_random['num_samples'].unique()

    ax.set_ylim([0.2, 1.0])
    ax.xaxis.set_ticks([c for i, c in enumerate(sample_sizes)
                               if i % 5 == 0])

    ax.xaxis.set_ticklabels(['25', '', '275', '', '525'])

    data_random = []

    for clf in ['transformer', 'setfit']:
        run = []
        for run_id in df_acc_random['run_id'].unique():
            run.append(df_acc_random.query(f'`classifier` == "{clf}" and run_id == {run_id}')['test_acc'].tolist())
        data_random.append(run)

    plot_learning_curve(
        ax,
        sample_sizes,
        np.array(data_random, dtype=float),
        ['BERT', 'SetFit'],
        strategy_name='random',
        show_uncertainty='tube-sd'
    )

    data_bt = []
    for clf in ['transformer', 'setfit']:
        run = []
        for run_id in df_acc_bt['run_id'].unique():
            run.append(df_acc_bt.query(f'`classifier` == "{clf}" and run_id == {run_id}')['test_acc'].tolist())
        data_bt.append(run)

    plot_learning_curve(
        ax,
        sample_sizes,
        np.array(data_bt, dtype=float),
        ['BERT', 'SetFit'],
        strategy_name='lc-bt',
        show_uncertainty='tube-sd'
    )

    data_gc = []
    for clf in ['transformer', 'setfit']:
        run = []
        for run_id in df_acc_gc['run_id'].unique():
            run.append(df_acc_gc.query(f'`classifier` == "{clf}" and run_id == {run_id}')['test_acc'].tolist())
        data_gc.append(run)

    plot_learning_curve(
        ax,
        sample_sizes,
        np.array(data_gc, dtype=float),
        ['BERT', 'SetFit'],
        strategy_name='gc',
        show_uncertainty='tube-sd'
    )

    # ax.set_xlabel('number of instances')
    # ax.set_ylabel('accuracy')
    ax.get_legend().remove()


def main():
    results = get_results('yb-coresets')

    results = check_for_duplicates(results)
    results = check_for_reproduciblity(results)

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.set_figheight(5)
    fig.set_figwidth(6*3)

    datasets = ['mr', 'ag-news', 'trec']
    for i in range(0, 3):
        plot_dataset(results, datasets[i], axs[i])
        axs[i].set_title(datasets[i], fontsize=20)

    plt.figlegend(
        axs[0].get_lines(),
        ['BERT / RS', 'SetFit / RS', 'BERT / BT', 'SetFit / BT', 'BERT / CS', 'SetFit / CS'],
        loc='upper center',
        prop={'size': 16},
        ncols=3
    )

    # fig.legend(loc='upper left', ncols=3)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.7)
    plt.savefig('plots.pdf')
    plt.show()


if __name__ == '__main__':
    mlflow.set_tracking_uri("file:///mnt/ceph/storage/data-tmp/current/yb63tadu/mlruns")
    main()
