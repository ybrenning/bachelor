from small_text.active_learner import PoolBasedActiveLearner

from active_learning_lab.experiments.active_learning.strategies import query_strategy_from_str


def get_initialized_active_learner(run_config, num_classes, train_set, test_set, x_ind_init):
    if isinstance(run_config.al_config.query_strategy, str):
        max_length = None \
            if 'max_length' not in run_config.dataset_config.dataset_kwargs \
            else run_config.dataset_config.dataset_kwargs['max_length']
        query_strategy = query_strategy_from_str(run_config.al_config.query_strategy,
                                                 run_config.al_config.query_strategy_kwargs,
                                                 num_classes,
                                                 max_length)
    else:
        raise NotImplementedError('Todo: Load query strategy from context')

    active_learner_type = run_config.al_config.active_learner_type
    if active_learner_type == 'default':
        active_learner = PoolBasedActiveLearner(
            run_config.classification_config.classifier_factory,
            query_strategy,
            train_set,
            reuse_model=run_config.al_config.reuse_model_across_queries)
    else:
        raise NotImplementedError(f'Invalid active_learner_type: {active_learner_type}')

    y_init = train_set[x_ind_init].y
    active_learner.initialize_data(x_ind_init, y_init, retrain=False)

    return active_learner, y_init
