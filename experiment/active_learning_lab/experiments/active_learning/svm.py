"""Example of a binary active learning text classification.
"""
import numpy as np

from small_text import (
    ConfidenceEnhancedLinearSVC,
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    SklearnClassifierFactory
)

from sklearn.feature_extraction.text import TfidfVectorizer
from small_text import SklearnDataset

from sklearn.datasets import fetch_20newsgroups

#from active_learning_lab.utils.experiment import set_random_seed

#set_random_seed(42)


def get_twenty_newsgroups_corpus(categories=['rec.sport.baseball', 'rec.sport.hockey']):

    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                               categories=categories)

    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                              categories=categories)

    return train, test


def get_train_test():
    return get_twenty_newsgroups_corpus()


def preprocess_data(train, test):
    vectorizer = TfidfVectorizer(stop_words='english')

    ds_train = SklearnDataset.from_arrays(train.data, train.target, vectorizer, train=True)
    ds_test = SklearnDataset.from_arrays(test.data, test.target, vectorizer, train=False)

    return ds_train, ds_test

from sklearn.metrics import f1_score


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred, train.y, average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    print('---')

from abc import ABC, abstractmethod
class Classifier(ABC):
    """Abstract base class for classifiers that can be used with the active learning components.
    """

    @abstractmethod
    def fit(self, train_set, weights=None):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : Dataset
            The dataset used for training the model.
        weights : np.ndarray[np.float32] or None, default=None
            Sample weights or None.
        """
        pass

    @abstractmethod
    def predict(self, data_set, return_proba=False):
        """Predicts the labels for each sample in the given dataset.

        Parameters
        ----------
        data_set : Dataset
            A dataset for which the labels are to be predicted.
        return_proba : bool, default=False
            If `True`, also returns a probability-like class distribution.
        """
        pass

    @abstractmethod
    def predict_proba(self, data_set):
        """Predicts the label distribution for each sample in the given dataset.

        Parameters
        ----------
        data_set : Dataset
            A dataset for which the labels are to be predicted.
        """
        pass

from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import is_multilabel
from small_text.utils.data import check_training_data
from small_text.utils.classification import empty_result, prediction_result
class SklearnClassifierNew(Classifier):
    """An adapter for using scikit-learn estimators.

    Notes
    -----
    The multi-label settings currently assumes that the underlying classifer returns a sparse
    matrix if trained on sparse data.
    """

    def __init__(self, model, num_classes, multi_label=False):
        """
        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            A scikit-learn estimator that implements `fit` and `predict_proba`.
        num_classes : int
            Number of classes which are to be trained and predicted.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        """
        if multi_label:
            self.model = OneVsRestClassifier(model)
        else:
            self.model = model
        self.num_classes = num_classes
        self.multi_label = multi_label

    def fit(self, train_set, weights=None):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : SklearnDataset
            The dataset used for training the model.
        weights : np.ndarray[np.float32] or None, default=None
            Sample weights or None.

        Returns
        -------
        clf : SklearnClassifier
            Returns the current classifier with a fitted model.
        """
        check_training_data(train_set, None, weights=weights)
        if self.multi_label and weights is not None:
            raise ValueError('Sample weights are not supported for multi-label SklearnClassifier.')

        y = train_set.y
        if self.multi_label and not is_multilabel(y):
            raise ValueError('Invalid input: Given labeling must be recognized as '
                             'multi-label according to sklearn.utils.multilabel.is_multilabel(y)')
        elif not self.multi_label and is_multilabel(y):
            raise ValueError('Invalid input: Given labeling is recognized as multi-label labeling '
                             'but the classifier is set to single-label mode')

        fit_kwargs = dict() if self.multi_label else dict({'sample_weight': weights})
        self.model.fit(train_set.x, y, **fit_kwargs)
        return self

    def predict(self, data_set, return_proba=False):
        """
        Predicts the labels for the given dataset.

        Parameters
        ----------
        data_set : SklearnDataset
            A dataset for which the labels are to be predicted.
        return_proba : bool, default=False
            If `True`, also returns a probability-like class distribution.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on multi-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32]
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        if len(data_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.model.predict_proba(data_set.x)

        return prediction_result(proba, self.multi_label, self.num_classes, enc=None,
                                 return_proba=return_proba)

    def predict_proba(self, data_set):
        """Predicts the label distribution for each sample in the given dataset.

        Parameters
        ----------
        data_set : SklearnDataset
            A dataset for which the labels are to be predicted.
        """
        if len(data_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False, return_proba=True)

        return self.model.predict_proba(data_set.x)

    def embed(self, dataset, embed_dim=5, pbar=None):
        _unused = pbar  # noqa:F841
        self.embeddings_ = np.random.rand(len(dataset), embed_dim)
        return self.embeddings_

def evaluate_multi_label(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    # https://github.com/scikit-learn/scikit-learn/issues/18611
    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred.toarray(), train.y.toarray(), average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test.toarray(),
                                                  test.y.toarray(), average='micro')))
    print('---')


class AbstractClassifierFactory(ABC):

    @abstractmethod
    def new(self):
        pass
from sklearn.base import BaseEstimator
from sklearn.base import clone
class SklearnClassifierFactoryNew(AbstractClassifierFactory):

    def __init__(self, base_estimator, num_classes, kwargs={}):
        """
        base_estimator : BaseEstimator
            A scikit learn estimator which is used as a template for creating new classifier objects.
        num_classes : int
            Number of classes.
        kwargs : dict
            Keyword arguments that are passed to the constructor of each classifier that is built by the factory.
        """
        if not issubclass(type(base_estimator), BaseEstimator):
            raise ValueError(
                'Given classifier template must be a subclass of '
                'sklearn.base.BaseEstimator. Encountered class was: {}.'
                .format(str(base_estimator.__class__))
            )

        self.base_estimator = base_estimator
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        """Creates a new SklearnClassifier instance.

        Returns
        -------
        classifier : SklearnClassifier
            A new instance of SklearnClassifier which is initialized with the given keyword args `kwargs`.
        """
        return SklearnClassifierNew(clone(self.base_estimator), self.num_classes, **self.kwargs)

    def __str__(self):
        return f'SklearnClassifierFactory(base_estimator={type(self.base_estimator).__name__}, ' \
               f'num_classes={self.num_classes}, kwargs={self.kwargs})'

def main(num_iterations=10):
    # Prepare some data: The data is a 2-class subset of 20news (baseball vs. hockey)
    text_train, text_test = get_train_test()
    train, test = preprocess_data(text_train, text_test)
    num_classes = 2

    # Active learning parameters
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactoryNew(clf_template, num_classes)

    from class_imbalances import ClassBalancer
    from strategies import GreedyCoreset
    query_strategy = ClassBalancer(GreedyCoreset())

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, labeled_indices, test, num_iterations)
    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):
    """
    This is the main loop in which we perform 10 iterations of active learning.
    During each iteration 20 samples are queried and then updated.

    The update step reveals the true label to the active learner, i.e. this is a simulation,
    but in a real scenario the user input would be passed to the update function.
    """
    # Perform 10 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[active_learner.indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    # Initialize the model. This is required for model-based query strategies.
    indices_pos_label = np.where(y_train == 1)[0]
    indices_neg_label = np.where(y_train == 0)[0]

    indices_initial = np.concatenate([np.random.choice(indices_pos_label, 10, replace=False),
                                      np.random.choice(indices_neg_label, 10, replace=False)],
                                     dtype=int)

    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for binary text classification.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
