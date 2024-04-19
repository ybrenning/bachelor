import os
import torch

import numpy as np

from pathlib import Path


def set_random_seed(seed, pytorch=True):
    # PYTHONHASHSEED and numpy seed have the smaller range (2**32-1)
    assert 0 <= seed <= 2**32-1

    os.environ['PYTHONHASHSEED'] = str(seed)
    if pytorch:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_data_dir():
    if 'DATA_DIR' in os.environ:
        base_dir = os.environ['DATA_DIR']
    else:
        base_dir = '.data/'
    return base_dir


def get_tmp_dir(tmp_base, folder_name):
    path = Path(tmp_base).joinpath(folder_name)
    if not path.exists():
        path.mkdir(parents=True)

    return str(path.absolute())


def get_configuration_id(config):

    if config['classifier']['classifier_name'] != 'transformer':
        return config['classifier']['classifier_name']

    classifier_kwargs = config['classifier']['classifier_kwargs']
    transformer_model = classifier_kwargs['transformer_model']

    if transformer_model == 'distilroberta-base':
        return 'distilroberta-base'
    elif transformer_model == 'google/electra-base-discriminator':
        return 'electra-base'
    elif transformer_model == 'google/electra-large-discriminator':
        return 'electra'
    elif transformer_model == 'bert-base-cased':
        return 'bert-base-cased'
    elif transformer_model == 'bert-base-uncased':
        return 'bert-base-uncased'
    elif transformer_model == 'bert-large-cased':
        return 'bert-large-cased'
    elif transformer_model == 'bert-large-uncased':
        return 'bert-large-uncased'

    raise ValueError(f'Unknown transformer_model {transformer_model}. Cannot derive a config id.')
