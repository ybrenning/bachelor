
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn.functional as F

    from torch import randperm
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

    from small_text.integrations.pytorch.classifiers.base import PytorchClassifier
    from small_text.integrations.pytorch.model_selection import Metric, PytorchModelSelection
    from small_text.integrations.pytorch.utils.data import dataloader, get_class_weights
    from small_text.integrations.transformers.classifiers.classification import (
        TransformerBasedClassification,
        TransformerBasedEmbeddingMixin
    )
    from small_text.integrations.transformers.datasets import TransformersDataset
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


class EfficientTransformerBasedClassification(TransformerBasedClassification):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.fit_count = 0

    def _initialize_optimizer_and_scheduler(self, optimizer, scheduler, num_epochs,
                                            sub_train, base_lr):
        steps = (len(sub_train) // self.mini_batch_size) \
                + int(len(sub_train) % self.mini_batch_size != 0)

        if optimizer is None:
            params, optimizer = self._default_optimizer(base_lr) \
                if optimizer is None else optimizer

        if scheduler == 'linear':
            try:
                from transformers import get_linear_schedule_with_warmup

                if self.fit_count <= 1:
                    total_steps = steps*num_epochs
                    warmup_steps = int(0.1 * total_steps)
                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=warmup_steps,
                                                                num_training_steps=total_steps - warmup_steps)
                else:
                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=0,
                                                                num_training_steps=steps * num_epochs)

            except ImportError:
                raise ValueError('Linear scheduler is only available when the transformers '
                                 'integration is installed ')

        elif scheduler is None:
            # constant learning rate
            scheduler = LambdaLR(optimizer, lambda _: 1)
        elif not isinstance(scheduler, _LRScheduler):
            raise ValueError(f'Invalid scheduler: {scheduler}')

        return optimizer, scheduler

    def fit(self, *args, **kwargs):

        if self.fit_count == 1:
            self.num_epochs = int(0.8 * self.num_epochs)
            print(f'New num epochs: {self.num_epochs}')

        result = super().fit(*args, **kwargs)
        self.fit_count += 1

        return result

    def __dict__(self):
        return dict()
