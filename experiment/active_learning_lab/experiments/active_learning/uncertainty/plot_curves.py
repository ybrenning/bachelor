import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from active_learning_lab.experiments.active_learning.postprocess.learning_curves import plot_learning_curve
from active_learning_lab.experiments.active_learning.uncertainty.settings import CONFIG_IDS, \
    DATASETS, QUERY_SIZES, QUERY_STRATEGIES
from active_learning_lab.utils.results import get_result_folders, check_for_duplicates, \
    assemble_df, check_for_reproduciblity


TPL_START = r"""
\begin{tabular}[t]{@{}l@{\hspace{4pt}}llp{10pt}cccccccp{10pt}ccccccc@{}}%
\\
\toprule
& & & \multicolumn{7}{@{}c@{}}{\bfseries Accuracy per Strategy}\\
\cmidrule{4-10} \cmidrule{12-18}\multicolumn{2}{@{}l@{}}{\bfseries Dataset / Model} & & ?? & PE & BT & BA & GC & CA & RS & &\\
\midrule
"""

TPL_END = r"""
\bottomrule
\end{tabular}%
"""

TITLE_FONTSIZE = 9


def main(experiment_name, output_folder):

    folders = get_result_folders(experiment_name)
    check_for_reproduciblity(folders)

    output_folder = Path(output_folder)
    if not output_folder.exists():
        raise ValueError('output folder does not exist!')

    df_all = assemble_df(folders, 'results.csv')
    df_all = df_all.set_index(['dataset_name', 'config_id', 'query_strategy', 'query_id'])
    check_for_duplicates(df_all)

    plot(df_all, output_folder.joinpath('learning-curves.pdf'))


def plot(df, output_file, plots_per_line=4):

    plt.clf()

    num_datasets = len(DATASETS)
    num_queries = len(df.reset_index()['query_id'].unique())  # including "query" 0
    num_runs = len(df['run_id'].unique())

    lines = (num_datasets) // plots_per_line + int(num_datasets % plots_per_line != 0)
    assert plots_per_line * lines >= num_datasets, "not enough space for plots and legend"

    fig, axes = plt.subplots(lines, plots_per_line, dpi=600, figsize=(9, 4.5), sharex=True, sharey=False)
    handles, labels = None, None

    for i, (d, d_label) in enumerate(DATASETS.items()):
        row = 0 if i < plots_per_line else i // plots_per_line
        col = i % plots_per_line

        target_ax = axes[row, col]

        data = np.empty((0, num_runs, num_queries), dtype=np.float64)
        labels = []

        for j, (q, q_label) in enumerate(QUERY_STRATEGIES.items()):
            try:
                gb_test_acc = df.loc[(d, 'bert-base-uncased', q)].sort_values(by='query_id').groupby('run_id')['test_acc']
                data_ = np.array([[d for _, d in gb_test_acc]])
                data = np.vstack((data, data_))
                labels.append(q_label)
            except KeyError:
                pass

        if data.shape[0] > 0:
            ax = plot_learning_curve(target_ax, QUERY_SIZES,
                                     data,
                                     labels,
                                     #linestyles=linestyles,
                                     show_uncertainty='tube-sd')

            target_ax.set_title(d_label, y=0.04, x=0.95, loc='right', fontsize=TITLE_FONTSIZE)
            target_ax.get_legend().set_visible(False)
            target_ax.yaxis.set_ticks([0.4, 0.6, 0.8, 1.0])

        if row == 0 and col == 0:
            handles, labels = axes[row, col].get_legend_handles_labels()

    labels, handles = zip(*sorted(zip(labels, handles)))
    legend = plt.legend(handles, labels, loc='lower right')
    legend.get_frame().set_linewidth(0.0)

    #f.suptitle(r'{\centering\LARGE{}Dataset: ' + last_dataset_name + r'}', usetex=True)

    axes[1, 3].remove()

    fig.text(0.5, 0, 'Labeled instances [%]', ha='center')
    fig.text(0, 0.5, 'Accuracy', va='center', rotation='vertical')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.046, hspace=0.07, left=0.075, bottom=0.125)

    plt.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment_name', type=str, help='')
    parser.add_argument('output_folder', type=str, help='')

    args = parser.parse_args()

    main(args.experiment_name, args.output_folder)
