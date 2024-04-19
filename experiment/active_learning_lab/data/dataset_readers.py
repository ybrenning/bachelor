import numpy as np
from pathlib import Path

from small_text.data import TextDataset
from small_text.utils.labels import list_to_csr

from active_learning_lab.data.dataset_abstractions import RawDataset
from active_learning_lab.data.dataset_preprocessing import (
    _get_huggingface_tokenizer,
    _text_to_bow,
    _text_to_huggingface,
    _text_to_tps)
from active_learning_lab.data.datasets import DataSets, DataSetType, DatasetReaderNotFoundException


def _read_trec(dataset_name: str, dataset_kwargs: dict, _: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    trec_dataset = datasets.load_dataset('trec')
    num_classes = 6

    label_col = 'coarse_label'

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    trec_dataset['train']['text'],
                                    trec_dataset['train'][label_col],
                                    trec_dataset['test']['text'],
                                    trec_dataset['test'][label_col],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return RawDataset(np.array(trec_dataset['train']['text']),
                          np.array(trec_dataset['train'][label_col]),
                          target_labels=np.arange(num_classes)), \
               RawDataset(np.array(trec_dataset['test']['text']),
                          np.array(trec_dataset['test'][label_col]),
                          target_labels=np.arange(num_classes))
    elif dataset_type == DataSetType.SETFIT:
        return TextDataset(np.array(trec_dataset['train']['text']),
                           np.array(trec_dataset['train'][label_col]),
                           target_labels=np.arange(num_classes)), \
               TextDataset(np.array(trec_dataset['test']['text']),
                           np.array(trec_dataset['test'][label_col]),
                           target_labels=np.arange(num_classes))
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(trec_dataset['train']['text'],
                            trec_dataset['train'][label_col],
                            trec_dataset['test']['text'],
                            trec_dataset['test'][label_col])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_tps(trec_dataset['train']['text'],
                            trec_dataset['train'][label_col],
                            trec_dataset['test']['text'],
                            trec_dataset['test'][label_col])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + dataset_name + '"')


def _read_agn(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
              classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    agn_dataset = datasets.load_dataset('ag_news')
    num_classes = 4

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    agn_dataset['train']['text'],
                                    agn_dataset['train']['label'],
                                    agn_dataset['test']['text'],
                                    agn_dataset['test']['label'],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return RawDataset(np.array(agn_dataset['train']['text']),
                          np.array(agn_dataset['train']['label']),
                          target_labels=np.arange(num_classes)), \
               RawDataset(np.array(agn_dataset['test']['text']),
                          np.array(agn_dataset['test']['label']),
                          target_labels=np.arange(num_classes))
    elif dataset_type == DataSetType.SETFIT:
        return TextDataset(np.array(agn_dataset['train']['text']),
                           np.array(agn_dataset['train']['label']),
                           target_labels=np.arange(num_classes)), \
               TextDataset(np.array(agn_dataset['test']['text']),
                           np.array(agn_dataset['test']['label']),
                           target_labels=np.arange(num_classes))
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_mr(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    mr_dataset = datasets.load_dataset('rotten_tomatoes')
    num_classes = 2

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    mr_dataset['train']['text'] + mr_dataset['validation']['text'],
                                    mr_dataset['train']['label'] + mr_dataset['validation']['label'],
                                    mr_dataset['test']['text'],
                                    mr_dataset['test']['label'],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW or dataset_type == DataSetType.SETFIT:
        return RawDataset(np.array(mr_dataset['train']['text'] + mr_dataset['validation']['text']),
                          np.array(mr_dataset['train']['label'] + mr_dataset['validation']['label']),
                          target_labels=np.arange(num_classes)), \
               RawDataset(np.array(mr_dataset['test']['text']),
                          np.array(mr_dataset['test']['label']),
                          target_labels=np.arange(num_classes))
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_tps(mr_dataset['train']['text'],
                            mr_dataset['train']['label'],
                            mr_dataset['test']['text'],
                            mr_dataset['test']['label'])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_dbp(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
              classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    dbp_dataset = datasets.load_dataset('dbpedia_14')
    num_classes = 14

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    dbp_dataset['train']['content'],
                                    dbp_dataset['train']['label'] ,
                                    dbp_dataset['test']['content'],
                                    dbp_dataset['test']['label'],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return RawDataset(dbp_dataset['train']['content'],
                          dbp_dataset['train']['label'],
                          target_labels=np.arange(num_classes)), \
               RawDataset(dbp_dataset['test']['content'],
                          dbp_dataset['test']['label'],
                          target_labels=np.arange(num_classes))
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


# TODO: this is a multi-label dataset
def _read_go_emotions(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    gem_dataset = datasets.load_dataset('go_emotions')
    num_classes = 28

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    gem_dataset['train']['text'] + gem_dataset['validation']['text'],
                                    list_to_csr(gem_dataset['train']['labels'] + gem_dataset['validation']['labels'],
                                                shape=(len(gem_dataset['train']['labels']) + len(gem_dataset['validation']['labels']), num_classes)),
                                    gem_dataset['test']['text'],
                                    list_to_csr(gem_dataset['test']['labels'], shape=(len(gem_dataset['test']['labels']), num_classes)),
                                    num_classes,
                                    int(dataset_kwargs['max_length']),
                                    is_multi_label=True)
    elif dataset_type == DataSetType.RAW or dataset_type == DataSetType.SETFIT:

        return RawDataset(np.array(gem_dataset['train']['text'] + gem_dataset['validation']['text']),
                          list_to_csr(np.array(gem_dataset['train']['labels'] + gem_dataset['validation']['labels']),
                                      shape=(len(gem_dataset['train']['labels']) + len(gem_dataset['validation']['labels']), num_classes)),
                          is_multi_label=True), \
               RawDataset(np.array(gem_dataset['test']['text']),
                          list_to_csr(np.array(gem_dataset['test']['labels']), shape=(len(gem_dataset['test']['labels']), num_classes)),
                          is_multi_label=True)
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_sst2(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import math
    import gluonnlp
    sst = gluonnlp.data.SST_2()
    num_classes = 2  # TODO: unverified

    # TODO: pass test (and for other datasets validation) split ratio
    TEST_SET_RATIO = 0.1

    test_set_size = int(math.ceil(len(sst) * TEST_SET_RATIO))
    indices = np.random.permutation(len(sst))

    train = [sst[i] for i in indices[test_set_size:]]
    test = [sst[i] for i in indices[:test_set_size]]

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    [item[0] for item in train],
                                    [item[1] for item in train],
                                    [item[0] for item in test],
                                    [item[1] for item in test],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return RawDataset([item[0] for item in train],
                          [item[1] for item in train]), \
               RawDataset([item[0] for item in test],
                          [item[1] for item in test])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_cr(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import math
    import gluonnlp
    cr = gluonnlp.data.CR()
    num_classes = 2  # TODO: unverified

    # TODO: pass test (and for other datasets validation) split ratio
    TEST_SET_RATIO = 0.1

    test_set_size = int(math.ceil(len(cr) * TEST_SET_RATIO))
    indices = np.random.permutation(len(cr))

    train = [cr[i] for i in indices[test_set_size:]]
    test = [cr[i] for i in indices[:test_set_size]]

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    [item[0] for item in train],
                                    [item[1] for item in train],
                                    [item[0] for item in test],
                                    [item[1] for item in test],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW or dataset_type == DataSetType.SETFIT:
        return RawDataset(np.array([item[0] for item in train]),
                          np.array([item[1] for item in train])), \
               RawDataset(np.array([item[0] for item in test]),
                          np.array([item[1] for item in test]))
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_imdb(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    imdb_dataset = datasets.load_dataset('imdb')
    num_classes = 2  # TODO: unverified

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    imdb_dataset['train']['text'],
                                    imdb_dataset['train']['label'],
                                    imdb_dataset['test']['text'],
                                    imdb_dataset['test']['label'],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return RawDataset(imdb_dataset['train']['text'],
                          imdb_dataset['train']['label']), \
               RawDataset(imdb_dataset['test']['text'],
                          imdb_dataset['test']['label'])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_yah(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
              classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import datasets
    yah_dataset = datasets.load_dataset('yahoo_answers_topics')
    num_classes = 10  # TODO: unverified

    def combine_text(dataset):
        questions_train = [title + ' ' + content
                           for title, content in zip(dataset['train']['question_title'],
                                                     dataset['train']['question_content'])]
        questions_train = [question + ' ' + answer
                           for question, answer in zip(questions_train,
                                                       dataset['train']['best_answer'])]
        questions_test = [title + ' ' + content
                          for title, content in zip(dataset['test']['question_title'],
                                                    dataset['test']['question_content'])]
        questions_test = [question + ' ' + answer
                          for question, answer in zip(questions_test,
                                                       dataset['test']['best_answer'])]

        return questions_train, questions_test

    train_text, test_text = combine_text(yah_dataset)
    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)

        return _text_to_huggingface(tokenizer,
                                    train_text,
                                    yah_dataset['train']['topic'],
                                    test_text,
                                    yah_dataset['test']['topic'],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(train_text,
                            yah_dataset['train']['topic'],
                            test_text,
                            yah_dataset['test']['topic'])
    elif dataset_type == DataSetType.RAW:
        return RawDataset(np.array(train_text),
                          np.array(yah_dataset['train']['topic'])), \
               RawDataset(np.array(test_text),
                          np.array(yah_dataset['test']['topic']))
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


def _read_subj(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
               classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    import math
    import gluonnlp
    subj = gluonnlp.data.SUBJ()
    num_classes = 2

    # TODO:
    TEST_SET_RATIO = 0.1

    test_set_size = int(math.ceil(len(subj) * TEST_SET_RATIO))
    indices = np.random.permutation(len(subj))

    train = [subj[i] for i in indices[test_set_size:]]
    test = [subj[i] for i in indices[:test_set_size]]

    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    [item[0] for item in train],
                                    [item[1] for item in train],
                                    [item[0] for item in test],
                                    [item[1] for item in test],
                                    num_classes,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW or dataset_type == DataSetType.SETFIT:
        return RawDataset(np.array([item[0] for item in train]),
                          np.array([item[1] for item in train]),
                          target_labels=np.arange(num_classes)), \
               RawDataset(np.array([item[0] for item in test]),
                          np.array([item[1] for item in test]),
                          target_labels=np.arange(num_classes))
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_tps([item[0] for item in train],
                            [item[1] for item in train],
                            [item[0] for item in test],
                            [item[1] for item in test])
    else:
        raise ValueError('Unsupported dataset type for dataset "' + str(dataset_name) + '"')


class OosEvalLoader():

    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def load_oos_eval(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(dir=self.data_dir) as tmp_dir:
            output_file, digest = self._load(tmp_dir)

            if digest != '36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0':
                raise ValueError()
            else:
                with open(output_file,'r') as f:
                    data = json.load(f)
                return data

    def _load(self, tmp_dir):
        import hashlib
        import requests

        output_file = Path(tmp_dir).joinpath('oos-eval.json')

        response = requests.get('https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json')

        md = hashlib.sha256()
        with open(output_file, 'wb') as fout:

            for chunk in response.iter_content(chunk_size=8096):
                if chunk:
                    fout.write(chunk)
                    md.update(chunk)

        return output_file, md.hexdigest()


def _read_oos_eval(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):
    loader = OosEvalLoader()

    result = loader.load_oos_eval()

    labels = np.unique(
        np.concatenate([
            [t[1] for t in result['train']],
            [t[1] for t in result['oos_train']],
            [t[1] for t in result['val']],
            [t[1] for t in result['oos_val']],
            [t[1] for t in result['test']],
            [t[1] for t in result['oos_test']]
        ])
    )

    label_dict = dict()
    for label in sorted(labels):
        if label != 'oos':
            label_dict[label] = len(label_dict)
    label_dict['oos'] = len(label_dict)

    # TODO: keep validation set
    train_text = np.concatenate([
        [t[0] for t in result['train']],
        [t[0] for t in result['oos_train']],
        [t[0] for t in result['val']],
        [t[0] for t in result['oos_val']]
    ])
    train_labels = np.concatenate([
        [label_dict[t[1]] for t in result['train']],
        [label_dict[t[1]] for t in result['oos_train']],
        [label_dict[t[1]] for t in result['oos_val']],
        [label_dict[t[1]] for t in result['val']]
    ])
    test_text = np.concatenate([
        [t[0] for t in result['test']],
        [t[0] for t in result['oos_test']]
    ])
    test_labels = np.concatenate([
        [label_dict[t[1]] for t in result['test']],
        [label_dict[t[1]] for t in result['oos_test']]
    ])
    if dataset_type == DataSetType.HUGGINGFACE:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_huggingface(tokenizer,
                                    train_text,
                                    train_labels,
                                    test_text,
                                    test_labels,
                                    int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.RAW:
        return (train_text, train_labels), \
               (test_text, test_labels)
    else:
        raise ValueError('Unsupported dataset type for dataset "' + dataset_name + '"')


DATASET_READERS = dict()
DATASET_READERS[DataSets.AG_NEWS] = _read_agn
DATASET_READERS[DataSets.CR] = _read_cr
DATASET_READERS[DataSets.MR] = _read_mr
DATASET_READERS[DataSets.TREC] = _read_trec
DATASET_READERS[DataSets.OOS_EVAL] = _read_oos_eval
DATASET_READERS[DataSets.SST2] = _read_sst2
DATASET_READERS[DataSets.GO_EMOTIONS] = _read_go_emotions
DATASET_READERS[DataSets.IMDB] = _read_imdb
DATASET_READERS[DataSets.YAH] = _read_yah
DATASET_READERS[DataSets.SUBJ] = _read_subj
DATASET_READERS[DataSets.DBP] = _read_dbp


def read_dataset(dataset_name: str, dataset_kwargs: dict, classifier_name: str,
                 classifier_kwargs: dict, dataset_type=None, data_dir='.data'):

    dataset = DataSets.from_str(dataset_name)
    if dataset in DATASET_READERS.keys():
        return DATASET_READERS[dataset](dataset_name, dataset_kwargs, classifier_name,
                                        classifier_kwargs, dataset_type, data_dir=data_dir)
    else:
        raise DatasetReaderNotFoundException(f'No reader registered for dataset \'{dataset_name}\'')

