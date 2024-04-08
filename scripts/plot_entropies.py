import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.1

format_names = {
    "mr": "MR",
    "ag-news": "AGN",
    "trec": "TREC",
    "gc": "Core-Set",
    "cb": "Class-Balanced Core-Set"
}

xs = [x for x in range(1, 21)]


def plot_entropy(ax, y, y_std, dataset, query_strategy):
    ax.set_ylim([0.8, 1.05])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])

    ax_lp = sns.lineplot(x=xs, y=y, ax=ax)

    lower, upper = y - y_std, y + y_std
    ax_lp.fill_between(xs, lower, upper, alpha=alpha)
    if dataset == "trec":
        ax_lp.set_xlabel('Query No.')

    if query_strategy == "gc":
        ax_lp.set_ylabel('Normalized Entropy')
    else:
        ax_lp.get_yaxis().set_visible(False)


def main():
    datasets = ['mr', 'ag-news', 'trec']
    query_strategies = ['gc', 'cb']
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

    plt.tight_layout()

    plt.savefig('entropy_plots.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
