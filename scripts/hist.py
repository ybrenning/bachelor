import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.1

xs = [x for x in range(1, 21)]


def plot_entropy(ax, y, y_std):
    ax.set_ylim([0.8, 1.05])

    ax_lp = sns.lineplot(x=xs, y=y, ax=ax)

    lower, upper = y - y_std, y + y_std
    ax_lp.fill_between(xs, lower, upper, alpha=alpha)
    ax_lp.set_xlabel('Query No.')
    ax_lp.set_ylabel('Entropy')


def main():
    datasets = ['mr', 'ag-news', 'trec']
    query_strategies = ['gc', 'cb']
    n = len(datasets)
    m = len(query_strategies)

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(12, 8))
    for i in range(0, n):
        for j in range(0, m):
            d = datasets[i]
            qs = query_strategies[j]
            values = np.load(f'data/entropy-plot-{qs}-{d}.npz')
            y = values['arr_0']
            y_std = values['arr_1']

            plot_entropy(axes[i, j], y, y_std)

    pad = 5

    for ax, col in zip(axes[0], query_strategies):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], datasets):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
