import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.1
datasets = ['mr', 'ag-news', 'trec']
query_strategies = ['gc', 'cb']

format_names = {
    "mr": "MR",
    "ag-news": "AGN",
    "trec": "TREC",
    "gc": "CS",
    "cb": "CB-CS"
}

xs = [x for x in range(1, 21)]

fontsize = 14


def plot_entropy(ax, y, y_std, dataset, query_strategy):
    ax.set_ylim([0.8, 1.05])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])

    ax_lp = sns.lineplot(x=xs, y=y, ax=ax)

    lower, upper = y - y_std, y + y_std
    ax_lp.fill_between(xs, lower, upper, alpha=alpha)

    if query_strategy == "gc":
        ax_lp.set_title(format_names[dataset], fontsize=fontsize)
        ax_lp.get_xaxis().set_visible(False)

    if dataset == "mr":
        ax_lp.set_ylabel('Normalized Entropy')
    else:
        ax_lp.get_yaxis().set_visible(False)


def old_plots():
    n = len(datasets)
    m = len(query_strategies)

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(6, 8))
    for i in range(0, n):
        for j in range(0, m):
            d = datasets[i]
            qs = query_strategies[j]
            values = np.load(f'data/entropy-plot-{qs}-{d}-transformer.npz')
            y = values['arr_0']
            y_std = values['arr_1']

            plot_entropy(axes[i, j], y, y_std, d, qs)

    pad = 5
    for i in range(3):
        for j in range(2):
            ax = axes[i, j]

            ax.set_xlabel('')
            ax.set_ylabel('')

    for ax, col in zip(axes[0], query_strategies):
        ax.annotate(format_names[col], xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], datasets):
        ax.annotate(
            format_names[row],
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad-pad, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center'
        )

    # fig.text(0.5, 0.0, 'common X', ha='center')
    # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    fig.supxlabel('Query Number', x=0.6, y=0.025)
    fig.supylabel('Normalized Entropy', x=0.025, y=0.51)

    plt.tight_layout()

    # plt.savefig('entropy_plots.pdf')
    plt.show()


def new_plots():
    n = len(query_strategies)
    m = len(datasets)

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(10, 6))

    for i in range(0, n):
        for j in range(0, m):
            qs = query_strategies[i]
            d = datasets[j]
            values = np.load(f'data/entropy-plot-{qs}-{d}-transformer.npz')
            y = values['arr_0']
            y_std = values['arr_1']

            plot_entropy(axes[i, j], y, y_std, d, qs)

    for i in range(0, n):
        for j in range(0, m):
            ax = axes[i, j]

            ax.set_xlabel('')
            ax.set_ylabel('')
            if j == m - 1:  # Only for the last column in each row
                if i == 0:
                    query_strategy = "gc"
                else:
                    query_strategy = "cb"

                ax.text(
                    1.05,
                    0.5,
                    format_names[query_strategy],
                    va='center',
                    ha='left',
                    transform=ax.transAxes,
                    fontsize=fontsize
                )

    # for ax, row in zip(axes[:, 2], query_strategies):
    #     ax.annotate(
    #         format_names[row],
    #         xy=(1, 0.5),
    #         xytext=(ax.yaxis.labelpad+pad, 0),
    #         xycoords=ax.yaxis.label, textcoords='offset points',
    #         size='large', ha='left', va='center'
    #     )

    fig.supxlabel('Query Number', x=0.5, y=0.025, fontsize=fontsize)
    fig.supylabel('Normalized Entropy', x=0.001, y=0.51, fontsize=fontsize)

    plt.tight_layout()

    plt.savefig('entropy_plots_new.pdf')
    # plt.show()


def main():
    # old_plots()
    new_plots()


if __name__ == '__main__':
    main()
