from dependency_injector import containers
from dependency_injector import providers

EXPERIMENT_NAME = 'minimal'

DEFAULT_CONFIG = {
    'general': {
        'runs': 5,
        'seed': 762671998,  # obtained via np.random.randint(2**32),
        'max_reproducibility': True,
    },
    'active_learner': {
        'active_learner_type': 'default',
        'num_queries': 10,
        'query_size': 8,
        'query_strategy': 'random',
        'query_strategy_kwargs': dict(),
        'initialization_strategy': 'srandom',
        'initialization_strategy_kwargs': dict({
            'num_instances': 20
        }),
        'initial_model_selection': 0,
        'validation_set_sampling': 'stratified'
    },
    'classifier': {
        'classifier_name': 'transformer',
        'validation_set_size': 0.1,
        'classifier_kwargs': dict({
            # 'multi_label': True,
            'transformer_model': 'prajjwal1/bert-medium',
        })
    }
}

TMP_BASE = "/var/tmp/cschroeder/active-learning-lab-v2"


def set_defaults(config: providers.Configuration) -> None:
    config.from_dict(DEFAULT_CONFIG)


def update_config(config: providers.Configuration) -> None:
    if config['classifier']['classifier_name'] == 'transformer':
        if config['dataset']['dataset_name'] == 'ag-news':
            config['dataset']['dataset_kwargs']['max_length'] = 128
            config['classifier']['classifier_kwargs'][
                'mini_batch_size'] = 32  # 128 ml / 256 mbs was too large
        elif config['dataset']['dataset_name'] == 'imdb':
            config['dataset']['dataset_kwargs']['max_length'] = 512
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 8
        elif config['dataset']['dataset_name'] == 'trec':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs'][
                'mini_batch_s ize'] = 32  # 64 ml / 512 mbs was too large
        elif config['dataset']['dataset_name'] == 'sst-2':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 64
        elif config['dataset']['dataset_name'] == 'go-emotions':
            config['dataset']['dataset_kwargs']['max_length'] = 64
            config['classifier']['classifier_kwargs']['mini_batch_size'] = 64


class Container(containers.DeclarativeContainer):
    pass
