import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMask(nn.Module):
    def __init__(self, max_size, ramp_size, shape=(1,)):
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape))

        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)

        mask = self.mask_template + self.current_val * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)

        if x.size(-1) < self._max_size:
            # The input could have been trimmed beforehand to save computation
            mask = mask[:, -x.size(-1):]

        x = x * mask.unsqueeze(0)  # Add unsqueeze to apply mask to each element in the batch

        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """This needs to be called after each update."""
        self.current_val.data.clamp_(0, 1)




class AdaptiveSpan(nn.Module):
    """Adaptive attention span for Transformerself.
    This module learns an attention span length from data for each
    self-attention head.

    Args:
        attn_span: maximum attention span
        adapt_span_loss: loss coefficient for the span length
        adapt_span_ramp: length of the masking ramp
        adapt_span_init: initial size ratio
        adapt_span_cache: adapt cache size to reduce memory usage
    """
    def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
                 adapt_span_init, adapt_span_cache, nb_heads, **kargs):
        super(AdaptiveSpan, self).__init__()
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                 ramp_size=adapt_span_ramp,
                                 init_val=adapt_span_init,
                                 shape=(nb_heads, 1, 1))

    def forward(self, attn, normalize=True):
        """Mask attention with the right span."""
        # Batch and head dimensions are merged together, so separate them first
        B = attn.size(0)  # Batch size
        M = attn.size(1)  # Block size
        attn = attn.reshape(B // self._nb_heads, self._nb_heads, M, -1)

        attn = self._mask(attn)
        if normalize:
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # Normalize so sum is 1

        attn = attn.view(B, M, -1)
        return attn

    def get_trim_len(self):
        """How much of memory can be trimmed to reduce computation."""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # Too fine granularity might be bad for memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        """Trim out unnecessary memory beforehand to reduce computation."""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            # Cache is too short! This happens when validation resumes after a lot of updates.
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe

    def get_cache_size(self):
        """Determine how long the cache should be."""
        if self._adapt_cache:
            trim_len = self.get_trim_len()
            # Give a buffer of 64 steps since a span might increase in future updates
            return min(self._max_span, self._max_span - trim_len + 64)
        else:
            return self._max_span

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()