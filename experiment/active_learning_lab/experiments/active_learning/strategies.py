from small_text.integrations.transformers.classifiers.classification import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.integrations.pytorch.query_strategies import (
    ExpectedGradientLength, ExpectedGradientLengthMaxWord)
from small_text.integrations.pytorch.query_strategies.strategies import BADGE
from small_text import BALD

from small_text.query_strategies import (QueryStrategy,
                                         LeastConfidence,
                                         RandomSampling,
                                         LightweightCoreset,
                                         PredictionEntropy,
                                         EmbeddingKMeans,
                                         DiscriminativeActiveLearning,
                                         CategoryVectorInconsistencyAndRanking,
                                         ContrastiveActiveLearning,
                                         SEALS)
from small_text.query_strategies.strategies import SubsamplingQueryStrategy, BreakingTies, EmbeddingBasedQueryStrategy

from .class_imbalances import ClassBalancer


def query_strategy_from_str(query_strategy_name, query_strategy_kwargs, num_classes, max_length=None):
    strategy = _get_query_strategy(query_strategy_name, num_classes, max_length)

    # TODO: None check should not be necessary
    if query_strategy_kwargs is not None and 'subsample' in query_strategy_kwargs:
        subsample_size = int(query_strategy_kwargs['subsample'])
        strategy = SubsamplingQueryStrategy(strategy, subsample_size)

    return strategy


import warnings
# TODO: weights must be within [0, 1]
import numpy as np

from sklearn.metrics import pairwise_distances
from small_text import BreakingTies
from small_text.query_strategies.strategies import EmbeddingBasedQueryStrategy


_DISTANCE_METRICS = ['cosine', 'euclidean']


def _check_coreset_size(x, n):
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')


def _cosine_distance(a, b, normalized=False):
    sim = np.matmul(a, b.T)
    if not normalized:
        sim = sim / np.dot(np.linalg.norm(a, axis=1)[:, np.newaxis],
                           np.linalg.norm(b, axis=1)[np.newaxis, :])
    return np.arccos(sim) / np.pi


def _euclidean_distance(a, b, normalized=False):
    _ = normalized
    return pairwise_distances(a, b, metric='euclidean')


def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(indices_unlabeled.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    return np.array(ind_new)


class GreedyCoreset(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)
        return greedy_coreset(embeddings, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
               f'normalize={self.normalize}, batch_size={self.batch_size})'

def greedy_coreset_balanced(x, y, clf, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(indices_unlabeled.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(5*n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    from small_text.data.sampling import _get_class_histogram
    # Get class distributions
    distributions = _get_class_histogram(y, clf.num_classes)

    distributions = distributions[y]

    from sklearn.preprocessing import normalize
    distributions = normalize(distributions)

    # Or like this?
    # min_val = np.min(distributions)
    # max_val = np.max(distributions)
    # distributions = (distributions - min_val) / (max_val - min_val)

    # To balance the classes, we want to give higher priority to those with less instances
    distributions = 1 - distributions
    ind_selected = np.argpartition(distributions, n)

    return np.array(ind_new)[ind_selected]

class GreedyCoresetCB(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)
        return greedy_coreset_balanced(embeddings, y, clf, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
               f'normalize={self.normalize}, batch_size={self.batch_size})'


# TODO: weights must be within [0, 1]
def ranked_weighted_greedy_coreset(x, embeddings_proba, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    weights : np.ndarray
        Array of weights.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n * 2):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):
            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    certainties = np.apply_along_axis(lambda x: BreakingTies._best_versus_second_best(
        x), 1, embeddings_proba[ind_new])

    # - argpartition ist effizienter als argsort, aber beides okay
    # - Fehler: sortiert wird auf der Submenge "uncertainties[ind_new]" anstatt auf "uncertainties"
    ind_selected = np.argpartition(certainties, n)[:n]

    # da ind_selected relativ zu ind_new ist, müssen wir das wieder über ind_new auflösen
    return np.array(ind_new)[ind_selected]


class RankedWeightedGreedyCoreset(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}, default='euclidean'
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool, default=True
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int, batch_size=100
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        # TODO: embeddings_proba should not be None
        # certainties = np.apply_along_axis(lambda x: BreakingTies._best_versus_second_best(
        #    x), 1, embeddings_proba)
        # breaking ties = 1 - margin
        # uncertainties = 1 - certainties
        return ranked_weighted_greedy_coreset(embeddings, embeddings_proba, indices_unlabeled, indices_labeled, n,
                                        distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'WeightedGreedyCoreset(distance_metric={self.distance_metric}, ' \
               f'normalize={self.normalize}, batch_size={self.batch_size})'


# TODO: weights must be within [0, 1]
def weighted_greedy_coreset(x, weights, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    weights : np.ndarray
        Array of weights.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(np.multiply(dists, 0.8) + np.multiply(weights[indices_unlabeled], 0.2))

        ind_new.append(index_new)

    return np.array(ind_new)


class WeightedGreedyCoreset(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}, default='euclidean'
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool, default=True
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int, batch_size=100
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        # TODO: embeddings_proba should not be None
        certainties = np.apply_along_axis(lambda x: BreakingTies._best_versus_second_best(
            x), 1, embeddings_proba)
        # breaking ties = 1 - margin
        uncertainties = 1 - certainties
        return weighted_greedy_coreset(embeddings, uncertainties, indices_unlabeled, indices_labeled, n,
                                        distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'WeightedGreedyCoreset(distance_metric={self.distance_metric}, ' \
               f'normalize={self.normalize}, batch_size={self.batch_size})'


class GreedyCoresetTSNE(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', perplexity=30, n_iter=1000)
        embeddings_reduced = tsne.fit_transform(embeddings)
        return greedy_coreset(embeddings_reduced, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
            f'normalize={self.normalize}, batch_size={self.batch_size})'

class GC_UMAP(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        import umap
        reducer = umap.UMAP( n_components=256, metric=self.distance_metric)

        embeddings_reduced = reducer.fit_transform(embeddings)
        return greedy_coreset(embeddings_reduced, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
            f'normalize={self.normalize}, batch_size={self.batch_size})'

class GC_TSNE_W(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', perplexity=30, n_iter=1000)
        embeddings_reduced = tsne.fit_transform(embeddings)

        certainties = np.apply_along_axis(lambda x: BreakingTies._best_versus_second_best(
            x), 1, embeddings_proba)
        # breaking ties = 1 - margin
        uncertainties = 1 - certainties
        return weighted_greedy_coreset(embeddings_reduced, uncertainties, indices_unlabeled, indices_labeled, n,
                                        distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
            f'normalize={self.normalize}, batch_size={self.batch_size})'


class GreedyCoreSetPCA_TSNE(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        embeddings_reduced = tsne.fit_transform(embeddings)
        return greedy_coreset(embeddings_reduced, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
               f'normalize={self.normalize}, batch_size={self.batch_size})'


def _get_query_strategy(str, num_classes, max_length):

    if str == 'lc-bt':
        return BreakingTies()
    elif str == 'lc-bt-s':
        return ClassBalancer(BreakingTies(), subsample_size=16_384)
    elif str == 'bald':
        return BALD()
    elif str == 'discr':
        # TODO: obtain factory from container
        transformer_model = TransformerModelArguments('bert-base-uncased')
        classifier_factory = TransformerBasedClassificationFactory(transformer_model, 2)
        return DiscriminativeActiveLearning(classifier_factory, 5)
    elif str == 'dal-eacl':
        from small_text import SetFitModelArguments, SetFitClassificationFactory
        # TODO: obtain factory from container
        model_args = SetFitModelArguments('sentence-transformers/paraphrase-mpnet-base-v2')
        classifier_factory = SetFitClassificationFactory(model_args,
                                                         num_classes,
                                                         classification_kwargs={'device': 'cuda'})
        return DiscriminativeActiveLearning(classifier_factory, 3)
    elif str == 'cal':
        return ContrastiveActiveLearning()
    elif str == 'gc':
        return GreedyCoreset()
    elif str == 'wrgc':
        return WeightedRepresentativeGreedyCoreset(num_classes)
    elif str == 'gc2':
        return GreedyCoresetLS()
    elif str == 'gc2-u':
        return GreedyCoresetLS(normalize=False)
    elif str == 'badge':
        return HubnessAnalyzingDecorator(ReverseNNDecorator(BADGE(num_classes)), num_classes)
    elif str == 'lc-lc':
        return LeastConfidence()
    elif str == 'km':
        return EmbeddingKMeans()
    elif str == 'km-ft':  # same as km, will just be handled differently in the experiments
        return EmbeddingKMeans()
    elif str == 'lc-ent':
        return PredictionEntropy()
    elif str == 'lc-ent--seals':
        return SEALS(PredictionEntropy(), k=10)
    elif str == 'gc--seals':
        return SEALS(GreedyCoreset5(), k=10)
    elif str == 'random':
        return RandomSampling()
    elif str == 'egl':
        return ExpectedGradientLength(num_classes)
    elif str in set(['coreset_lightweight', 'lightweight_coreset', 'lwcs']):
        return LightweightCoreset()
    elif str == 'cvirs':
        return CategoryVectorInconsistencyAndRanking()
    elif str == 'gc-tsne':
        return GreedyCoresetTSNE()
    elif str == 'gc-pca-tsne':
        return GreedyCoreSetPCA_TSNE()
    elif str == 'wgc':
        return WeightedGreedyCoreset()
    elif str == 'rwgc':
        return RankedWeightedGreedyCoreset()
    elif str == 'cb':
        return ClassBalancer(GreedyCoreset())
    elif str == 'wgc-tsne':
        return GC_TSNE_W()
    elif str == 'ugc':
        return GC_UMAP()
    else:
        raise ValueError('Unknown query strategy string: ' + str)
