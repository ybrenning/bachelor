import numpy as np

from scipy.sparse import csr_matrix

from small_text.data.datasets import SklearnDataset, DatasetView, TextDataset, TextDatasetView
from small_text.data.sampling import multilabel_stratified_subsets_sampling

from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
from small_text.integrations.pytorch.utils.data import _get_class_histogram
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.utils.data import list_length
from small_text.data.sampling import balanced_sampling, stratified_sampling

from active_learning_lab.data.dataset_abstractions import RawDataset


# TODO: can this moved into the library?
def get_x_y(dataset):

    if isinstance(dataset, (RawDataset, SklearnDataset, TransformersDataset, PytorchTextClassificationDataset, DatasetView)):
        x = dataset.x
        y = dataset.y
    else:
        raise ValueError('NotImplemented')

    return x, y


def get_class_histogram(y, num_classes, normalize=True):
    ind, counts = np.unique(y, return_counts=True)
    ind_set = set(ind)

    histogram = np.zeros(num_classes)
    for i, c in zip(ind, counts):
        if i in ind_set:
            histogram[i] = c

    if normalize:
        return histogram / histogram.sum()

    return histogram.astype(int)


def get_num_class(dataset):
    if issubclass(type(dataset), (RawDataset, PytorchTextClassificationDataset,
                                  SklearnDataset, TransformersDataset,
                                  TextDataset, TextDatasetView)):
        if dataset.multi_label:
            return len(np.unique(dataset.y.indices))
        else:
            return len(np.unique(dataset.y))
    elif isinstance(dataset, list) and len(dataset[0]) == 3:
        raise NotImplementedError('Is this still required?')
        #return len(np.unique([d[2] for d in dataset]))
    elif isinstance(dataset, tuple):
        raise NotImplementedError('Is this still required?')
        #return len(np.unique(dataset[1]))

    raise ValueError('Unsopported type: ' + str(type(dataset)))



def get_validation_set(y, classifier_name, strategy='balanced', validation_set_size=0.1,
                       multilabel_strategy='labelsets'):

    if classifier_name == 'svm':
        return None

    if validation_set_size == 0.0:
        return None

    n_samples = int(validation_set_size * list_length(y))

    if strategy == 'balanced':
        return balanced_sampling(y, n_samples=n_samples)
    elif strategy == 'stratified':
        if isinstance(y, csr_matrix):
            if multilabel_strategy == 'labelsets':
                return multilabel_stratified_subsets_sampling(y, n_samples=n_samples)
            else:
                raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
        else:
            return stratified_sampling(y, n_samples=n_samples)

    raise ValueError(f'Invalid strategy: {strategy}')


def get_class_weights(y, num_classes, eps=1e-8):
    label_counter = _get_class_histogram(y, num_classes, normalize=False)
    pos_weight = np.ones(num_classes, dtype=float)
    num_samples = len(y)
    for c in range(num_classes):
        pos_weight[c] = (num_samples - label_counter[c]) / (label_counter[c] + eps)

    if num_classes == 2:
        pos_weight[pos_weight.argmin()] = 1.0

    return pos_weight
