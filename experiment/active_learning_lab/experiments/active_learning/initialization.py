import numpy as np
from typing import Dict, List

from small_text.data import Dataset, TextDataset
from small_text.data.sampling import _get_class_histogram, _random_sampling


def get_initial_indices(train_set: Dataset, train_text: TextDataset, initialization_strategy: str,
                        initialization_strategy_kwargs: dict, num_samples: int):

    if initialization_strategy == 'random':
        from small_text.initialization import random_initialization
        x_ind_init = random_initialization(train_set, n_samples=num_samples)
    elif initialization_strategy == 'srandom':
        from small_text.initialization import random_initialization_stratified
        y_train = train_set.y
        x_ind_init = random_initialization_stratified(y_train, n_samples=num_samples)
    elif initialization_strategy == 'balanced':
        from small_text.initialization import random_initialization_balanced
        y_train = train_set.y
        x_ind_init = random_initialization_balanced(y_train, n_samples=num_samples)
    elif initialization_strategy == 'keywords':
        if not 'keywords' in initialization_strategy_kwargs:
            raise ValueError('Insufficient arguments for initialization strategy "keywords": '
                             'keywords not found in initialization_strategy_kwargs')

        x_ind_init = sample_by_keywords(train_text, initialization_strategy_kwargs['keywords'], n_samples=num_samples)
    else:
        raise ValueError('Invalid initialization strategy: ' + initialization_strategy)

    return x_ind_init


def sample_by_keywords(train_text: TextDataset, keywords_per_class: Dict[int, List[str]], n_samples=25):
    inv_index = _build_inverted_index(train_text)

    num_samples_in_dataset = len(train_text)
    num_classes = len(keywords_per_class)
    num_classes_present = len(np.unique(train_text.y))

    scores = np.zeros((num_samples_in_dataset, num_classes), dtype=int)
    for cls_idx in keywords_per_class.keys():
        for keyword in keywords_per_class[cls_idx]:
            if keyword in inv_index:
                indices_with_keyword = inv_index[keyword]
                scores[indices_with_keyword, cls_idx] += 1

    # TODO: generalize this for num_classes > 2
    assert num_classes == 2
    y_pred_by_keywords = scores.argmax(axis=1)
    scores = scores.max(axis=1) - scores.min(axis=1)

    indices = []
    samples_per_class = _get_samples_required_per_class(n_samples, num_classes)

    for c, num_samples_c in enumerate(samples_per_class):
        class_indices = np.argwhere(y_pred_by_keywords == c)[:, 0]
        assert class_indices.shape[0] > 0

        indices_best = np.argpartition(-scores[class_indices], num_samples_c)[:num_samples_c]
        indices.extend(np.arange(y_pred_by_keywords.shape[0])[class_indices[indices_best]])

        # indices = _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y_pred_by_keywords)
        # indices_resolved = np.arange(len(train_text))[indices_at_least_one_keyword][indices]
    print(indices)

    for idx in indices:
        print(train_text.y[idx], train_text.x[idx])

    return np.array(indices)


def _build_inverted_index(train_text: TextDataset, lowercase=True):
    from tokenizers.pre_tokenizers import Whitespace

    pre_tokenizer = Whitespace()
    inv_index = {}

    for i, text in enumerate(train_text.x):
        for token, _ in pre_tokenizer.pre_tokenize_str(text):
            if lowercase:
                token = token.lower()
            inv_index[token] = inv_index.get(token, []) + [i]

    for token in inv_index.keys():
        inv_index[token] = list(set(inv_index[token]))

    return inv_index


def _get_samples_required_per_class(n_samples, n_classes):
    samples_per_class = [n_samples // n_classes] * n_classes

    remainder = n_samples // n_classes
    if  remainder != 0:
        for _ in range(remainder):
            random_class = np.random.randint(0, n_classes)
            samples_per_class[random_class] += 1

    return samples_per_class
