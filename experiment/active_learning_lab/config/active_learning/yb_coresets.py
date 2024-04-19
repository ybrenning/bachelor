"""
import socket

from dependency_injector import containers
from dependency_injector import providers


EXPERIMENT_NAME = 'yb-coresets'


DEFAULT_CONFIG = {
    'general': {
        'runs': 3,  # 10 for an improved version of this experiment
        'seed': 1003,
        'max_reproducibility': True,
    },
    'active_learner': {
        'active_learner_type': 'default',
        'active_learner_kwargs': dict({
            'reuse_model': False
        }),
        'num_queries': 10,
        'query_size': 25,
        'query_strategy': 'random',
        'query_strategy_kwargs': dict(),
        'initialization_strategy': 'srandom',
        'initialization_strategy_kwargs': dict({
            'num_instances': 25
        }),
        'initial_model_selection': 0,
        'validation_set_sampling': 'stratified',
        'stopping_criteria': []
    },
    'classifier': {
        'classifier_name': 'transformer',
        'validation_set_size': 0.1,
        'classifier_kwargs': dict({
            'multi_label': False
        })
    }
}


TMP_BASE = "/var/tmp/yb63tadu/active-learning-lab-v2"


def set_defaults(config: providers.Configuration, override_args: dict) -> None:
    new_dict = DEFAULT_CONFIG

    override_args['active_learner']['query_strategy_kwargs']['subsample'] = 20000

    if override_args['active_learner']['query_strategy'] == 'cal':
        if override_args['classifier']['classifier_name'] == 'transformer':
            # use cls embedding for transformer
            override_args['active_learner']['query_strategy_kwargs']['embed_kwargs'] = {'embedding_method': 'cls'}

    config.from_dict(new_dict)


def update_config(config: providers.Configuration) -> None:
    if config['classifier']['classifier_name'].startswith('transformer'):
        if 'transformer_model' not in config['classifier']['classifier_kwargs']:
            config['classifier']['classifier_kwargs']['transformer_model'] = 'bert-large-uncased'


        if config['dataset']['dataset_name'] == 'mr':
            config['dataset']['dataset_kwargs']['max_length'] = 60
        elif config['dataset']['dataset_name'] == 'trec':
            config['dataset']['dataset_kwargs']['max_length'] = 40
        elif config['dataset']['dataset_name'] == 'ag-news':
            config['dataset']['dataset_kwargs']['max_length'] = 60
        elif config['dataset']['dataset_name'] == 'cr':
            config['dataset']['dataset_kwargs']['max_length'] = 50
        elif config['dataset']['dataset_name'] == 'subj':
            config['dataset']['dataset_kwargs']['max_length'] = 50


class Container(containers.DeclarativeContainer):
    pass
"""

from dependency_injector import containers
from dependency_injector import providers


EXPERIMENT_NAME = 'yb-coresets'


DEFAULT_CONFIG = {
    'general': {
        'runs': 5,  # 10 for an improved version of this experiment
        'seed': 1003,
        'max_reproducibility': True,
    },
    'active_learner': {
        'active_learner_type': 'default',
        'active_learner_kwargs': dict({
            'reuse_model': False
        }),
        'num_queries': 20,
        'query_size': 25,
        'query_strategy': 'random',
        'query_strategy_kwargs': dict(),
        'initialization_strategy': 'srandom',
        'initialization_strategy_kwargs': dict({
            'num_instances': 25
        }),
        'initial_model_selection': 0,
        'validation_set_sampling': 'stratified',
        'stopping_criteria': [
            #'kappa_k90', 'kappa_k95', 'kappa_k99',
            'ouncertainty_005', 'ouncertainty_010', 'ouncertainty_015',
            'cchange_005', 'cchange_010', 'cchange_015'
        ],
        'query_strategy_kwargs': {}
    },
    'classifier': {
        'classifier_name': 'transformer',
        'validation_set_size': 0.1,
        'classifier_kwargs': dict({
            'multi_label': False
        })
    }
}


TMP_BASE = "/var/tmp/yb63tadu/active-learning-lab-v2"


CLASSIFIER_TRANSFORMER = {
    'classifier_name': 'transformer',
    'validation_set_size': 0.1,
    'classifier_kwargs': dict({
        'multi_label': False,
        'lr': 0.00002,
        'scheduler': 'slanted',
        'layerwise_gradient_decay': 0.975,
        'mini_batch_size': 16
    })
}


CLASSIFIER_SETFIT = {
    'classifier_name': 'setfit',
    'validation_set_size': 0.1,
    'classifier_kwargs': dict({
        'multi_label': False,
        'trainer_kwargs': {
            'num_epochs': 1,
            'learning_rate': 0.00002
        },
        'mini_batch_size': 16
    })
}


def set_defaults(config: providers.Configuration, override_args: dict) -> None:
    new_dict = DEFAULT_CONFIG

    override_args['active_learner']['query_strategy_kwargs']['subsample'] = 20000

    if override_args['active_learner']['query_strategy'] == 'cal':
        if override_args['classifier']['classifier_name'] == 'transformer':
            # use cls embedding for transformer
            override_args['active_learner']['query_strategy_kwargs']['embed_kwargs'] = {'embedding_method': 'cls'}

    if override_args['classifier']['classifier_name'].startswith('transformer'):
        new_dict['classifier'] = CLASSIFIER_TRANSFORMER

        if override_args['dataset']['dataset_name'] == 'ag-news':
            new_dict['classifier']['classifier_kwargs']['num_epochs'] = 50
        else:
            new_dict['classifier']['classifier_kwargs']['num_epochs'] = 15
    elif override_args['classifier']['classifier_name'] == 'setfit':
        new_dict['classifier'] = CLASSIFIER_SETFIT
        new_dict['classifier']['classifier_kwargs']['transformer_model'] = 'sentence-transformers/paraphrase-mpnet-base-v2'
        new_dict['classifier']['classifier_kwargs']['trainer_kwargs']['num_epochs'] = 1

        new_dict['classifier']['classifier_kwargs']['model_kwargs'] = {'cache_dir': TMP_BASE}

    if override_args['dataset']['dataset_name'] in ['ag-news']:
        new_dict['active_learner']['query_strategy_kwargs']['subsample'] = 20_000
        new_dict['classifier']['classifier_kwargs']['mini_batch_size'] = 12

    config.from_dict(new_dict)


def update_config(config: providers.Configuration) -> None:
    if config['classifier']['classifier_name'].startswith('transformer'):
        config['classifier']['classifier_kwargs']['early_stopping_no_improvement'] = 5
        config['classifier']['classifier_kwargs']['memory_fix'] = False
        config['classifier']['classifier_kwargs']['model_selection'] = True

        if 'transformer_model' not in config['classifier']['classifier_kwargs']:
            config['classifier']['classifier_kwargs']['transformer_model'] = 'bert-base-uncased'


        if config['dataset']['dataset_name'] == 'mr':
            config['dataset']['dataset_kwargs']['max_length'] = 60
        elif config['dataset']['dataset_name'] == 'trec':
            config['dataset']['dataset_kwargs']['max_length'] = 40
        elif config['dataset']['dataset_name'] == 'ag-news':
            config['dataset']['dataset_kwargs']['max_length'] = 60
        elif config['dataset']['dataset_name'] == 'cr':
            config['dataset']['dataset_kwargs']['max_length'] = 50
        elif config['dataset']['dataset_name'] == 'subj':
            config['dataset']['dataset_kwargs']['max_length'] = 50


class Container(containers.DeclarativeContainer):
    pass

