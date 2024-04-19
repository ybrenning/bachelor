import numpy as np

from small_text.data.sampling import stratified_sampling

from active_learning_lab.data.dataset_readers import DataSetType, read_dataset
from active_learning_lab.utils.data import get_x_y, get_class_histogram


class DatasetLoader(object):

    def __init__(self, data_dir='.data'):
        self.data_dir = data_dir
        self.subsampled_indices = dict()

    def load_dataset(self, dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                     classifier_kwargs: dict, dataset_type=None):

        dataset_type_expected = get_dataset_type(classifier_name, dataset_kwargs, dataset_type)
        # TODO: train, test, valid
        train, test = read_dataset(dataset_name, dataset_kwargs,
                                   classifier_name, classifier_kwargs, dataset_type=dataset_type_expected,
                                   data_dir=self.data_dir)

        if 'subsample' in dataset_kwargs:
            if dataset_name not in self.subsampled_indices:
                self.subsampled_indices[dataset_name] = stratified_sampling(
                    train.y,
                    n_samples=int(dataset_kwargs['subsample']))

            if dataset_type_expected == DataSetType.RAW:
                # TODO: convenience method to make a copy
                train.x = [train.x[i] for i in self.subsampled_indices[dataset_name]]
            else:
                # TODO: convenience method to make a copy
                train._data = [train._data[i] for i in self.subsampled_indices[dataset_name]]

        if 'target_distribution' in dataset_kwargs:
            target_distribution = dataset_kwargs['target_distribution']
            if not isinstance(target_distribution, list):
                raise ValueError('Target distribution must be a list')

            train = subsample_to_target_distribution(train, target_distribution)

        return train, test


def subsample_to_target_distribution(dataset, target_distribution):
    """Subsamples from the given dataset to obtain a subset with a specific target distribution.

    If `target_distribution` is of type numpy.ndarray[float]:
    Computes the true distribution of the given dataset and adjusts the subsample relatively
    per class, i.e., for each class the ndarray contains a number 0.0 < n <= 1.0 which defines
    what percentage of this class is retained.

    If `target_distribution` is of type numpy.ndarray[int]:
    Randomly draws the given number of samples for each class.

    Parameters
    ----------
    dataset : small_text.data.Dataset
        Input dataset from which a subsample is drawn.
    target_distribution : numpy.ndarray[float] or numpy.ndarray[int]
        Target distribution of shape (num_classes,)

    Returns
    -------
    indices : numpy.ndarray[int]
        Indices relative to dataset.x which comprise the newly subsampled subset.
    """

    if np.issubdtype(target_distribution, float):

        assert 0.0 <= target_distribution.all() <= 1.0, ""
        y = dataset.y

        num_classes = np.unique(y).shape[0]

        encountered_distribution = get_class_histogram(y, num_classes, normalize=False)
        if encountered_distribution.shape[0] != target_distribution.shape[0]:
            raise ValueError('encountered distribution and target distribution differ in their '
                             'number of classes')

        sample_distribution = encountered_distribution * target_distribution
        sample_distribution = sample_distribution.round().astype(int)

    elif np.issubdtype(target_distribution, int):

        sample_distribution = target_distribution

    else:
        raise ValueError('Invalid target distribution')

    y_indices = np.arange(len(y))

    indices = np.empty((0,), dtype=np.int)
    for c in range(sample_distribution.shape[0]):
        new_indices = np.random.choice(y_indices[y == c], sample_distribution[c], replace=False)
        indices = np.append(indices, new_indices)

    return dataset[indices]


def get_dataset_type(classifier_name, dataset_kwargs, dataset_type):

    if 'dataset_type' in dataset_kwargs:
        dataset_type_expected = DataSetType.from_str(dataset_kwargs['dataset_type'])
    elif dataset_type is not None:
        if isinstance(dataset_type, DataSetType):
            dataset_type_expected = dataset_type
        else:
            dataset_type_expected = DataSetType.from_str(dataset_type)
    else:
        dataset_type_expected = get_dataset_type_for_classifier(classifier_name)

    return dataset_type_expected


def get_dataset_type_for_classifier(classifier_name):

    if classifier_name == 'lgbm' or classifier_name == 'svm':
        return DataSetType.BOW
    elif classifier_name == 'kimcnn':
        return DataSetType.TENSOR_PADDED_SEQ
    elif classifier_name == 'transformer' or classifier_name == 'transformer-mc' or classifier_name == 'transformer-func':
        return DataSetType.HUGGINGFACE
    elif classifier_name in ['setfit', 'setfit-sc']:
        return DataSetType.SETFIT

    raise ValueError(f'No dataset type defined for classifier_name={classifier_name}')
