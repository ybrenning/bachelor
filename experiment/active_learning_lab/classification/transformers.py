import copy
import logging
import warnings
import types
import torch

from torch.nn.modules import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

from small_text.classifiers.factories import AbstractClassifierFactory
from small_text.integrations.transformers.classifiers.classification import FineTuningArguments, TransformerModelArguments, TransformerBasedClassification

from torch.optim import AdamW

from active_learning_lab.classification.extensions import predict_proba_mc
from active_learning_lab.pytorch.lr_scheduler import LinearSchedulerWithUnfreezing


class TransformerBasedFunctionalClassificationFactory(AbstractClassifierFactory):

    def __init__(self, transformer_model_args, num_classes, kwargs={}):
        """
        Parameters
        ----------
        transformer_model_args : TransformerModelArguments
            Name of the sentence transformer model.
        num_classes : int
            Number of classes.
        kwargs : dict
            Keyword arguments which will be passed to `TransformerBasedClassification`.
        """
        self.transformer_model_args = transformer_model_args
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):
        from active_learning_lab.classification.transformers_ext.bert_func import BertForSequenceClassificationFunctional

        kwargs_new = copy.deepcopy(self.kwargs)
        for key in ['transformer_model', 'transformer_config', 'transformer_tokenizer']:
            if key in kwargs_new:
                del kwargs_new[key]

        transformer_model = TransformerModelArguments(
            self.kwargs['transformer_model'],
            self.kwargs.get('transformer_tokenizer', None),
            self.kwargs.get('transformer_config', None)
        )

        clf = TransformerBasedClassification(transformer_model,
                                             self.num_classes,
                                             **kwargs_new)
        return clf


class HuggingfaceTransformersClassificationFactory(AbstractClassifierFactory):

    def __init__(self, classifier_name, num_classes, kwargs={}):
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.kwargs = kwargs

    def new(self):

        kwargs_new = copy.deepcopy(self.kwargs)
        for key in ['transformer_model', 'transformer_config', 'transformer_tokenizer',
                    'acl22_optimizer']:
            if key in kwargs_new:
                del kwargs_new[key]

        fine_tuning_args = None

        # TODO: logic
        if 'scheduler' in self.kwargs and self.kwargs['scheduler'] == 'slanted':
            gradual_unfreezing = self.kwargs.get('gradual_unfreezing', -1)
            fine_tuning_args = FineTuningArguments(self.kwargs['lr'],
                                                   self.kwargs['layerwise_gradient_decay'],
                                                   gradual_unfreezing=gradual_unfreezing)

            if 'layerwise_gradient_decay' in kwargs_new:
                del kwargs_new['layerwise_gradient_decay']
            if 'gradual_unfreezing' in kwargs_new:
                del kwargs_new['gradual_unfreezing']
            del kwargs_new['scheduler']
        else:
            # TODO: check if layerwise_gradient_decay is still set, delete and print warn
            pass

        transformer_model = TransformerModelArguments(
            self.kwargs['transformer_model'],
            self.kwargs.get('transformer_tokenizer', None),
            self.kwargs.get('transformer_config', None)
        )

        classifier_cls = self.kwargs.get('classifier_cls', TransformerBasedClassification)
        if 'classifier_cls' in kwargs_new:
            del kwargs_new['classifier_cls']

        clf = classifier_cls(transformer_model,
                             num_classes=self.num_classes,
                             fine_tuning_arguments=fine_tuning_args,
                             **kwargs_new)

        if 'acl22_optimizer' in self.kwargs:
            logging.info('acl22_optimizer extra activated')
            from active_learning_lab.classification.transformers_ext.acl22_specifics import _default_optimizer
            clf._default_optimizer = types.MethodType(_default_optimizer, clf)

        clf._initialize_optimizer_and_scheduler = types.MethodType(_initialize_optimizer_and_scheduler, clf)
        clf.initialize_transformer = types.MethodType(_initialize_transformer, clf)
        clf._perform_model_selection = types.MethodType(_perform_model_selection, clf)

        return clf


# https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/13
# https://github.com/pytorch/pytorch/issues/7415
def _perform_model_selection(self, optimizer, model_selection):
    model_selection_result = model_selection.select()
    if model_selection_result is not None:
        self.model.load_state_dict(torch.load(model_selection_result.model_path, map_location='cpu'))
        optimizer_path = model_selection_result.model_path.with_suffix('.pt.optimizer')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))


from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _initialize_transformer(self, cache_dir):

    self.config = AutoConfig.from_pretrained(
        self.transformer_model.config,
        num_labels=self.num_classes,
        cache_dir=cache_dir,
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.transformer_model.tokenizer,
        cache_dir=cache_dir,
    )

    # Suppress "Some weights of the model checkpoint at [model name] were not [...]"-warnings
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    self.model = AutoModelForSequenceClassification.from_pretrained(
        self.transformer_model.model,
        from_tf=False,
        config=self.config,
        cache_dir=cache_dir,
    )

    try:
        import torch
        from torch import _dynamo
        #torch._dynamo.config.suppress_errors = True
        # self.model.bert = torch.compile(self.model.bert)
        # https://github.com/pytorch/pytorch/commit/e071d72f3c9ba7e58ddb4cfcf0f4563e0e522bcf
        # https://discuss.pytorch.org/t/torch-dynamo-exc-unsupported-tensor-backward/169246/2
        #self.model.bert = _dynamo.optimize('aot_eager')(self.model.bert)
    except:
        warnings.warn('torch dynamo not found: could not compile model')

    transformers_logging.set_verbosity(previous_verbosity)


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
            total_steps = steps * self.num_epochs
            warmup_steps = min(0.1 * total_steps, 100)

            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps - warmup_steps)
        except ImportError:
            raise ValueError('Linear scheduler is only available when the transformers '
                             'integration is installed ')

    elif scheduler is None:
        # constant learning rate
        scheduler = LambdaLR(optimizer, lambda _: 1)
    elif not isinstance(scheduler, _LRScheduler):
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return optimizer, scheduler


from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn.modules import MSELoss




def forward_bert(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    features = outputs[0]
    averaged_output = torch.mean(features[:, 1:, :], dim=1)
    averaged_output = self.dropout(averaged_output)

    logits = self.classifier(averaged_output)

    loss = None
    if labels is not None:
        if self.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def __DUPLICATE_DO_NOT_USE_initialize_optimizer_and_scheduler(self, optimizer, scheduler, fine_tuning_arguments,
                                        base_lr, params, model, sub_train):

    steps = (len(sub_train) // self.mini_batch_size) \
            + int(len(sub_train) % self.mini_batch_size != 0)

    base_model = getattr(self.model_args, self.model_args.base_model_prefix)
    num_layers = len(base_model.encoder.layer)

    layer_prefix = f'{base_model}.encoder.layer'

    # <sun paper>
    groups = [(f'{layer_prefix}.{i}.', base_lr*0.9**(num_layers-i-1)) for i in range(num_layers)]

    decay_params = []
    no_decay_params = []

    for l, lr in groups:
        decay_params.append(
            {
                'params': [p for n, p in model.named_parameters() if
                           not 'bias' in n and n.startswith(l)],
                'weight_decay_rate': 0.01, 'lr': lr
            }
        )
        no_decay_params.append(
            {
                'params': [p for n, p in model.named_parameters() if
                           'bias' in n and n.startswith(l)],
                'weight_decay_rate': 0.0, 'lr': lr
            }
        )


    group_all_params = [
            {'params': [p for n, p in model.named_parameters() if
                        not 'bias' in n and not n.startswith(f'{layer_prefix}.')],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        'bias' in n and not n.startswith(f'{layer_prefix}.')],
             'weight_decay_rate': 0.0},
        ]

    params = group_all_params + decay_params + no_decay_params
    # </sun paper>

    # TOOD: dont override if optimizer is set
    optimizer = AdamW(params, lr=base_lr, eps=1e-8)

    total_steps = steps * self.num_epochs

    scheduler = LinearSchedulerWithUnfreezing(optimizer,
                                              num_warmup_steps=0.1 * total_steps,
                                              num_training_steps=0.9 * total_steps,
                                              gradual_unfreezing=num_layers / 3
                                              )

    """if scheduler == 'linear':
        total_steps = steps * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.1*total_steps,
                                                    num_training_steps=steps*self.num_epochs)
    elif scheduler == 'linear-unfreezing':

        total_steps = steps*self.num_epochs

        scheduler = LinearSchedulerWithUnfreezing(optimizer,
                                                  num_warmup_steps=0.1*total_steps,
                                                  num_training_steps=0.9*total_steps,
                                                  gradual_unfreezing=num_layers/3
                                                  )
    else:
        raise ValueError(f'Invalid scheduler: {scheduler}')"""

    return optimizer, scheduler


class LinearSchedulerWithUnfreezing(LambdaLR):

    def __init__(self, optimizer, num_warmup_steps, num_training_steps,
                 gradual_unfreezing=-1, last_epoch=-1):

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps))
            )

        self.optimizer = optimizer

        self.gradual_unfreezing = gradual_unfreezing
        if gradual_unfreezing > 0:
            self.unfreezing_steps = int(num_training_steps / gradual_unfreezing)

        self.last_epoch = last_epoch
        super(LinearSchedulerWithUnfreezing, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

    def get_lr(self):
        iteration = self.last_epoch - 1

        lrs = [lr for lr in self.base_lrs]

        if self.gradual_unfreezing:
            lrs = self._unfrozen_lrs(lrs, iteration)

        return lrs

    def _unfrozen_lrs(self, lrs, iteration):
        num_always_unfrozen = len(lrs) - self.gradual_unfreezing
        lrs = [lr if num_always_unfrozen + i + iteration >= self.gradual_unfreezing * self.unfreezing_steps - 1 else 0 for i, lr in enumerate(lrs)]
        return lrs
