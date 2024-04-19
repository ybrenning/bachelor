import numpy as np
import pandas as pd

from pathlib import Path

from active_learning_lab.experiments.active_learning.uncertainty.settings import CONFIG_IDS, DATASETS, QUERY_STRATEGIES
from active_learning_lab.utils.results import get_result_folders, check_for_duplicates, \
    assemble_df, check_for_reproduciblity


TPL_START = r"""
\begin{tabular}[t]{@{}l@{\hspace{4pt}}lp{10pt}ccccccc}%
\\
\toprule
& & & \multicolumn{7}{@{}c@{}}{\bfseries Accuracy per Strategy}\\
\cmidrule{4-10} \multicolumn{2}{@{}l@{}}{\bfseries Dataset / Model} & & ?? & PE & BT & BA & GC & CA & RS\\
\midrule
"""

TPL_END = r"""
\bottomrule
\end{tabular}%
"""


def main(experiment_name, output_folder):

    folders = get_result_folders(experiment_name)
    check_for_reproduciblity(folders)

    output_folder = Path(output_folder)
    if not output_folder.exists():
        raise ValueError('output folder does not exist!')

    collect_and_write_results(folders, output_folder)


def collect_and_write_results(folders, output_folder):

    df_agg, df_auc_agg = aggregate_dfs(folders)

    metric = 'auc_test_acc'
    write_table(df_auc_agg, output_folder.joinpath('table-auc-agg.tex'), metric, TPL_START, TPL_END)

    metric = 'test_acc'
    write_table(df_agg, output_folder.joinpath('table-acc-agg.tex'), metric, TPL_START, TPL_END)

    metric = 'test_ece_10'
    write_table(df_agg, output_folder.joinpath('table-ece-agg.tex'), metric, TPL_START, TPL_END)


def aggregate_dfs(folders):

    df_all = assemble_df(folders, 'results.csv')
    check_for_duplicates(df_all)
    final_query = df_all['query_id'].max()

    df_auc = assemble_df(folders, 'auc.csv')
    df_agg = df_all.groupby(['dataset_name', 'config_id', 'query_strategy', 'query_id']) \
        .agg([np.mean, np.std])

    df_agg.columns = df_agg.columns.to_flat_index()
    df_agg.columns = [tup[0] + '_' + tup[1] for tup in df_agg.columns]

    df_auc_agg = df_auc.groupby(['dataset_name', 'config_id', 'query_strategy']) \
        .agg([np.mean, np.std])

    df_auc_agg.columns = df_auc_agg.columns.to_flat_index()
    df_auc_agg.columns = [tup[0] + '_' + tup[1] for tup in df_auc_agg.columns]

    idx = pd.IndexSlice

    # only consider final query in this table
    df_agg = df_agg.loc[idx[:, :, :, final_query], :]
    assert df_agg.shape[0] == df_auc_agg.shape[0]
    df_agg = df_agg.reset_index().groupby(['dataset_name', 'config_id', 'query_strategy']).first()

    return df_agg, df_auc_agg


def write_table(df_agg, output_file, metric, tpl_start, tpl_end):
    with open(output_file, 'w+', encoding='utf-8') as f:
        f.write(tpl_start)

        idx = pd.IndexSlice

        for i, (d, d_label) in enumerate(DATASETS.items()):
            f.write(f'% {d_label}\n')
            try:
                df_sub = df_agg.loc[idx[d], :]
                write_row(d, d_label, df_sub, f, idx, metric)
            except KeyError:
                for i, (c, c_label) in enumerate(CONFIG_IDS.items()):
                    if i == 0:
                        f.write('\\multirow{3}{*}{' + d_label + '} & ' + c_label + ' & & ')
                    else:
                        f.write(' & ' + c_label + ' & & ')

                    for j, (q, q_label) in enumerate(QUERY_STRATEGIES.items()):
                        if j > 0:
                            f.write(' &')
                        f.write(' --')

                    f.write('\\\\\n')

            if i != len(DATASETS) - 1:
                f.write('\\midrule\n')

        f.write(tpl_end)


def write_row(d, d_label, df_sub, f, idx, metric):

    for i, (c, c_label) in enumerate(CONFIG_IDS.items()):
        if i == 0:
            f.write('\\multirow{3}{*}{' + d_label + '} & ' + c_label + ' & & ')
        else:
            f.write(' & ' + c_label + ' & & ')

        """if i == 0:
            num_configs = len(config_ids)
            f.write(f'    %')
            for q, q_label in query_strategies.items():
                f.write(f' {q}')
            f.write('\n')
            f.write(f'    \hline'
                    f'\multirow{{{num_configs}}}{{*}}{{{d_label}}} & {c_label} & ')
        else:
            f.write(f'     & {c_label}  & ')"""
        for j, (q, q_label) in enumerate(QUERY_STRATEGIES.items()):
            print(q_label)
            if j > 0:
                f.write(' &')

            try:
                result = df_sub.loc[idx[c, q, :], :]
                if result.shape[0] == 0:
                    f.write(' --')
                # elif d in set(['cr', 'mr', 'subj']) and q == 'lc-bt':
                #    # same as lc-ent
                #    f.write(' --')
                else:
                    assert result.shape[0] == 1
                    mean = result[metric + '_mean'][0]
                    std = result[metric + '_std'][0]
                    #f.write(f' \\num{{{mean:.3f}+-{std:.3f}}}')
                    f.write(f' {mean:.4f}')

            except KeyError:
                f.write(' --')

        f.write('\\\\\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment_name', type=str, help='')
    parser.add_argument('output_folder', type=str, help='')

    args = parser.parse_args()

    main(args.experiment_name, args.output_folder)
