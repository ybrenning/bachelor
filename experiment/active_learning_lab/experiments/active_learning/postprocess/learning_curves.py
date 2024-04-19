import numpy as np
import seaborn as sns

from scipy.stats import norm

strat_name_to_repr = {
    "random": "Random Sampling",
    "lc-bt": "Breaking-Ties",
    "gc": "Core-Set",
    "gc-tsne": "Core-Set with TSNE",
    "wgc": "Weighted Core-Set",
    "rwgc": "Re-ranked Core-Set",
    "cb": "Class-Balanced Core-Set",
    "ugc": "Core-Set with UMAP"
}
def plot_learning_curve(ax, collection_sizes, scores, labels, strategy_name, palette,
                        show_uncertainty='bars', alpha=0.1):
    """
    Plots the learning curve for multiple query strategies a single dataset.

    Parameters
    ----------
    ax : axis

    collection_sizes : list of integer
        A list of integers that contains the number of documents at the measured point.

    scores : 3-dimensional array (dimensions k x n x m)
        An array that contains k different query strategies for which n measurements have been performed
        on subsamples of a document collection of size m.
        This array is usually the result of a cross-validation.

    labels : list of strings
        A label for each metric used in the `scores` parameter.

    Returns
    -------
    ax : Axes
        Returns the Axes object with the learning curve plot.
    """
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    for i in range(0, len(labels)):

      mean = np.mean(scores[i], axis=0)

      ax_lp = sns.lineplot(x=collection_sizes, y=mean, label=strat_name_to_repr[strategy_name], ax=ax,
                           color=palette)

      if show_uncertainty:
          std = np.std(scores[i], axis=0)
          if show_uncertainty == 'bars':
              ax.errorbar(collection_sizes, mean, yerr=std)
          elif show_uncertainty == 'tube-sd':
              # TODO: name this. sdtube?
              lower, upper = mean - std, mean + std
              ax_lp.fill_between(collection_sizes, lower, upper, alpha=alpha, color=palette)
          else:
              # TODO: name this. tube?
              lower, upper = _ci(mean, std)
              ax_lp.fill_between(collection_sizes, lower, upper, alpha=alpha, color=palette)

    # Comment this out if you don't care about line style as this is very bad code
    if strategy_name == 'random':
        ax_lp.lines[0].set_linestyle('dashed')
    elif strategy_name == 'lc-bt':
        ax_lp.lines[1].set_linestyle('dotted')
    elif strategy_name == 'gc':
        ax_lp.lines[2].set_linestyle('dashdot')
    elif strategy_name == 'gc-tsne':
        ax_lp.lines[3].set_linestyle('solid')
    elif strategy_name == 'wgc':
        ax_lp.lines[4].set_linestyle('solid')
    elif strategy_name == 'rwgc':
        ax_lp.lines[5].set_linestyle('solid')
    elif strategy_name == 'cb':
        ax_lp.lines[6].set_linestyle('solid')


    return ax


def plot_inset(ax, axins, collection_sizes, scores, labels, strategy_name, palette,
                        show_uncertainty='bars', alpha=0.1):

    for i in range(0, len(labels)):

        mean = np.mean(scores[i], axis=0)
        ax_lp = sns.lineplot(x=collection_sizes, y=mean, label=strat_name_to_repr[strategy_name], ax=axins,
                             color=palette)

        if show_uncertainty:
            std = np.std(scores[i], axis=0)
            if show_uncertainty == 'bars':
                ax.errorbar(collection_sizes, mean, yerr=std)
            elif show_uncertainty == 'tube-sd':
                # TODO: name this. sdtube?
                lower, upper = mean - std, mean + std
                ax_lp.fill_between(collection_sizes, lower, upper, alpha=alpha, color=palette)
            else:
                # TODO: name this. tube?
                lower, upper = _ci(mean, std)
                ax_lp.fill_between(collection_sizes, lower, upper, alpha=alpha, color=palette)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    if strategy_name == 'random':
        ax_lp.lines[-1].set_linestyle('dashed')
    elif strategy_name == 'lc-bt':
        ax_lp.lines[-1].set_linestyle('dotted')
    elif strategy_name == 'gc':
        ax_lp.lines[-1].set_linestyle('dashdot')
    elif strategy_name == 'gc-tsne':
        ax_lp.lines[-1].set_linestyle('solid')
    elif strategy_name == 'wgc':
        ax_lp.lines[-1].set_linestyle('solid')
    elif strategy_name == 'rwgc':
        ax_lp.lines[-1].set_linestyle('solid')
    elif strategy_name == 'cb':
        ax_lp.lines[-1].set_linestyle('solid')

    axins.get_legend().remove()

def _ci(mean, sigma):
    # the first run tends to yield the same scores, which leads to nan values here
    lower, upper = norm.interval(0.95, loc=mean, scale=sigma)

    lower, upper = np.nan_to_num(lower), np.nan_to_num(upper)
    return lower, upper
