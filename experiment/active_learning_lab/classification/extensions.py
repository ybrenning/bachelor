import numpy as np

import torch
import torch.nn.functional as F  # noqa: N812

from functools import partial

from small_text.integrations.pytorch.utils.data import dataloader
from small_text.utils.classification import empty_result
from small_text.utils.context import build_pbar_context


# TODO: this can be removed (but is currently still referenced in UST)
def predict_proba_mc(self, test_set, T=30):  # T=30 in (Mukherjee, 2020)
    if len(test_set) == 0:
        return empty_result(self.multi_label, self.num_classes, return_prediction=False,
                            return_proba=True)

    self.model_args.eval()
    modules = dict({name: module for name, module in self.model_args.named_modules()})
    for name, mod in modules.items():
        if 'dropout' in name:
            mod.train()

    test_iter = dataloader(test_set.data, self.mini_batch_size, self._create_collate_fn(),
                           train=False)

    predictions = []
    logits_transform = torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)

    with torch.no_grad(), build_pbar_context('tqdm', tqdm_kwargs={'total': len(test_set)}) as pbar:
        for text, masks, *_ in test_iter:
            batch_size = text.shape[0]
            vector_len = text.shape[1]

            text, masks = text.to(self.device), masks.to(self.device)
            text, masks = text.repeat(1, T).resize(batch_size * T, vector_len), \
                          masks.repeat(1, T).resize(batch_size * T, vector_len)
            outputs = self.model_args(text, attention_mask=masks)
            pred = logits_transform(outputs.logits)
            pred = pred.unsqueeze(dim=1).resize(batch_size, T, self.num_classes)
            predictions += pred.detach().to('cpu').tolist()

            pbar.update(batch_size)
            #predictions += pred.mean(dim=1).detach().to('cpu').tolist()
            #var += torch.var(pred, unbiased=False, dim=1).detach().to('cpu').tolist()
            del text, masks

    for name, mod in modules.items():
        if 'dropout' in name:
            mod.eval()

    return np.array(predictions)
