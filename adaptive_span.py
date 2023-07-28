import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.

    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        mask = self.mask_template + self.current_val * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self._max_size:
            # the input could have been trimmed beforehand to save computation
            mask = mask[:, :, -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val.data.clamp_(0, 1)


class ImportanceScorer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImportanceScorer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        importance_scores = torch.sigmoid(self.fc2(x))
        return importance_scores


class AdaptiveSpan(nn.Module):
    def __init__(self, attn_span, nb_heads, input_size, hidden_size, **kargs):
        nn.Module.__init__(self)
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                 ramp_size=adapt_span_ramp,
                                 init_val=adapt_span_init,
                                 shape=(nb_heads, 1, 1))

        self.importance_scorer = ImportanceScorer(input_size, hidden_size)

    def forward(self, attn, normalize=True):
        """mask attention with the right span and dynamic threshold"""
        B = attn.size(0)  # batch size
        M = attn.size(1)  # block size

        # Calculate importance scores using the learnable importance scorer
        importance_scores = self.importance_scorer(attn.view(B * self._nb_heads, M, -1))
        importance_scores = importance_scores.view(B, self._nb_heads, M)

        dynamic_factor = torch.mean(importance_scores, dim=1, keepdim=True)
        self._mask.current_val = dynamic_factor

        # Apply dynamic masking based on importance scores
        masked_attn = self._mask(attn)

        if normalize:
            masked_attn = masked_attn / (masked_attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1

        masked_attn = masked_attn.view(B, M, -1)

        return masked_attn

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()

