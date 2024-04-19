import math
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class PolynomialDecayWithWarmup(LambdaLR):

    def __init__(self, optimizer, num_warmup_steps, num_total_steps, max_learning_rate, end_learning_rate,
                 last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return max_learning_rate / num_warmup_steps * (current_step + 1)
            elif current_step >= num_warmup_steps and current_step < num_total_steps:
                gamma = math.log(end_learning_rate / max_learning_rate) * (current_step - num_warmup_steps+1) / (num_total_steps - num_warmup_steps+1)
                return max_learning_rate * math.exp(gamma)
            else:
                return end_learning_rate

            return end_learning_rate

        self.last_epoch = last_epoch
        super(PolynomialDecayWithWarmup, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class ConstantLR(LambdaLR):

    def __init__(self, optimizer, last_epoch=-1):
        def lr_lambda(_current_step: int):
            return 1.0

        self.last_epoch = last_epoch
        super(ConstantLR, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class LinearSchedulerWithUnfreezing(LambdaLR):

    def __init__(self, optimizer, num_warmup_steps, num_training_steps,
                 gradual_unfreezing=-1, unfreezing_steps=None, last_epoch=-1):

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0



        self.optimizer = optimizer

        self.gradual_unfreezing = gradual_unfreezing
        self.unfreezing_steps = unfreezing_steps

        self.last_epoch = last_epoch
        super(LinearSchedulerWithUnfreezing, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

    def get_lr(self):

        # "last_epoch-1" because get_lr gets called once during init and thereby increases last_epoch
        iteration = self.last_epoch - 1

        lrs = [lr for lr in self.base_lrs]

        #if self.gradual_unfreezing:
        #    lrs = self._unfrozen_lrs(lrs, iteration)

        #print(lrs)

        return lrs

    def _unfrozen_lrs(self, lrs, iteration):
        num_always_unfrozen = len(lrs) - self.gradual_unfreezing
        lrs = [lr if num_always_unfrozen + i + iteration >= self.gradual_unfreezing * self.unfreezing_steps - 1 else 0 for i, lr in enumerate(lrs)]
        return lrs


class LinearSchedulerWithUnfreezing2(LambdaLR):

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
        super(LinearSchedulerWithUnfreezing2, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

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
