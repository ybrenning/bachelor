import numpy as np
import numpy.typing as npt

from typing import List, Union

from small_text.data.sampling import _get_class_histogram

from small_text.query_strategies.base import constraints, ClassificationType
from small_text.query_strategies.strategies import QueryStrategy
#from active_learning_lab.experiments.active_learning.strategies import greedy_coreset

from sklearn.metrics import pairwise_distances
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

# The naming here is kept general (distributions and categories), but this is currently used to create distributions
#  over the number of classes
def _sample_distribution(num_samples: int,
                         source_distribution: npt.NDArray[np.uint],
                         ignored_values: List[int] = []):
    """Return a balanced sample from the given `source_distribution` of size ``num-samples`. The sample is represented
    in the form of an empirical categorial frequency distribution (i.e. a histogram). It is built iteratively, and
    prefers the category currently having the smallest number of samples.

    Parameters
    ----------
    num_samples : int
        Number of samples that the resulting distribution has.
    source_distribution : np.ndarray[int]
        A source frequency distribution in the shape of (num_values,) where num_values is the number of possible values
        for the source_distribution.
    ignored_values : list of int
        List of values (indices in the interval [0, `source_distribution.shape[0]`]) that should be ignored.

    Returns
    -------
    output_distribution : np.ndarray[int]
        A new distribution, which is  whose categories are less than or equal to the source distribution.
    """

    num_classes = source_distribution.shape[0]
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_values)])

    new_distribution = np.zeros((num_classes,), dtype=int)
    for _ in range(num_samples):
        distribution_difference = (new_distribution - source_distribution)[active_classes]
        minima = np.where(distribution_difference == distribution_difference.min())[0]

        # Sample the class which occurs the least. In the case of a tie, the decision is random.
        if minima.shape[0] == 1:
            new_distribution[active_classes[minima[0]]] += 1
        else:
            sampled_minimum_index = np.random.choice(minima, 1)[0]
            new_distribution[active_classes[sampled_minimum_index]] += 1

    return new_distribution


def _get_rebalancing_distribution(num_samples, num_classes, y, y_pred, ignored_classes=[]):
    current_class_distribution = _get_class_histogram(y, num_classes)
    predicted_class_distribution = _get_class_histogram(y_pred, num_classes)

    number_per_class_required_for_balanced_dist = current_class_distribution.max() - current_class_distribution

    number_per_class_required_for_balanced_dist[list(ignored_classes)] = 0

    # Balancing distribution: When added to current_class_distribution, the result is balanced.
    optimal_balancing_distribution = current_class_distribution.max() - current_class_distribution
    target_distribution = _sample_distribution(num_samples,
                                               optimal_balancing_distribution,
                                               ignored_values=ignored_classes)

    # balancing_distribution:
    balancing_distribution = np.zeros((num_classes,), dtype=int)
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_classes)])

    for c in active_classes:
        if predicted_class_distribution[c] < target_distribution[c]:
            # adapt the balancing distribution so that it can be sampled
            balancing_distribution[c] = predicted_class_distribution[c]
        else:
            balancing_distribution[c] = target_distribution[c]

    # The predicted labels does not have enough classes so that a sample with the desired balancing distribution
    # cannot be provided. Try to fill the remainder with other samples from "active classes" instead.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        current_class_distribution += balancing_distribution

        free_active_class_samples = []
        for c in active_classes:
            class_indices = np.argwhere(y_pred == c)[:, 0]
            if class_indices.shape[0] > current_class_distribution[c]:
                free_active_class_samples.extend([c] * (class_indices.shape[0] - current_class_distribution[c]))

        np.random.shuffle(free_active_class_samples)
        for c in free_active_class_samples[:remainder]:
            balancing_distribution[c] += 1
            current_class_distribution[c] += 1

    # When not enough samples can be taken from the active classes, we fall back to using all classes.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        free_ignored_class_samples = []
        for i, count in enumerate(predicted_class_distribution - balancing_distribution):
            if count > 0:
                free_ignored_class_samples.extend([i] * count)

        np.random.shuffle(free_ignored_class_samples)
        for c in free_ignored_class_samples[:remainder]:
            balancing_distribution[c] += 1

    return balancing_distribution


def create_query_distribution(n, n_active_classes):
    n_per_class = n // n_active_classes
    remainder = n % n_active_classes
    query_dist = np.linspace(n_per_class, n_per_class, n_active_classes)
    query_dist[-1] += remainder
    return query_dist.astype(int)

class ClassBalancer(QueryStrategy):
    """A query strategy that tries to draw instances so that the new class distribution of the labeled pool
    is moved towards a (more) balanced distribution. For this, it first partitions instances by their
    predicted class and then applies a base query strategy. Based on the per-class query results, the
    instances are sampled so that the new class distribution is more balanced.

    Since the true labels are unknown, this strategy is a best effort approach and is not guaranteed
    to improve the distribution's balance.

    To reduce the cost of the initial predictions, which are required for the class-based partitioning,
    a random subsampling parameter is available.

    .. note ::
       The sampling mechanism is tailored to single-label classification.

    .. versionadded:: 2.0.0
    """

    def __init__(self, base_query_strategy: QueryStrategy, ignored_classes: List[int] = [],
                 subsample_size: Union[int, None] = None):
        """
        base_query_strategy : QueryStrategy
            A base query strategy which operates on the subsets partitioned by predicted class.
        subsample_size : int or None
            Draws a random subsample before applying the strategy if not `None`.
        """
        self.base_query_strategy = base_query_strategy
        self.ignored_classes = ignored_classes
        self.subsample_size = subsample_size
        self.distance_metric = "euclidean"
        self.normalize = True

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        if self.subsample_size is None or self.subsample_size > indices_unlabeled.shape[0]:
            indices = indices_unlabeled
        else:
            indices_all = np.arange(indices_unlabeled.shape[0])
            indices_subsample = np.random.choice(indices_all,
                                                 self.subsample_size,
                                                 replace=False)
            indices = indices_unlabeled[indices_subsample]

        return self._query_class_balanced(clf, dataset, indices, indices_labeled, y, n)

    def _query_class_balanced(self, clf, dataset, indices, indices_labeled, y, n):
        y_pred = clf.predict(dataset[indices])

        target_distribution = _get_rebalancing_distribution(n,
                                                            clf.num_classes,
                                                            y,
                                                            y_pred,
                                                            ignored_classes=self.ignored_classes)

        active_classes = np.array([i for i in range(clf.num_classes) if i not in set(self.ignored_classes)])

        embeddings = clf.embed(dataset)

        indices_balanced = []
        indices_labeled_tmp = indices_labeled

        n_active_classes = len(active_classes)
        # Determine size of coreset per class
        query_dist = create_query_distribution(n, n_active_classes)

        for idx, c in enumerate(active_classes):
            class_indices = indices[np.argwhere(y_pred == c)[:, 0]]
            class_indices = np.setdiff1d(class_indices, indices_labeled_tmp)

            if target_distribution[c] > 0:
                # class_reduced_indices = np.append(indices[class_indices], indices_labeled)

                from sklearn.preprocessing import normalize
                embeddings = normalize(embeddings, axis=1)

                queried_indices = greedy_coreset(
                    embeddings, class_indices, indices_labeled_tmp, query_dist[c],
                    distance_metric=self.distance_metric, normalized=self.normalize)

                indices_labeled_tmp = np.append(indices_labeled_tmp, class_indices[queried_indices])

                indices_balanced.extend(class_indices[queried_indices].tolist())
            else:
                redistributed = create_query_distribution(query_dist[c], n_active_classes - 1)

                zeros_array = np.zeros_like(query_dist)
                padded = np.concatenate((zeros_array[:len(query_dist) - len(redistributed)], redistributed))
                query_dist = query_dist + padded

        return np.array(indices_balanced)

    def __str__(self):
        return f'ClassBalancer(base_query_strategy={self.base_query_strategy}, ' \
               f'ignored_classes={self.ignored_classes}, ' \
               f'subsample_size={self.subsample_size})'
