#!/usr/bin/env python3

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
        x = x * mask.unsqueeze(1).unsqueeze(1)  # Broadcast mask to match x's shape
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
        """this needs to be called after each update"""
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
        nb_heads: number of attention heads
    """

    def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
                 adapt_span_init, adapt_span_cache, nb_heads, **kargs):
        nn.Module.__init__(self)
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                  ramp_size=adapt_span_ramp,
                                  init_val=adapt_span_init,
                                  shape=(nb_heads, 1, 1))

    def calculate_important_scores(self, x):
        # Calculate important scores for tokens based on the input x.
        important_scores = torch.mean(x, dim=-1, keepdim=True)
        return important_scores


        
    def calculate_dynamic_factors(self, important_scores):
        dynamic_factors = torch.sigmoid(self._mask.current_val.unsqueeze(-1)) * important_scores
        return dynamic_factors

    def calculate_dynamic_threshold(self, dynamic_factors):
        # Calculate the dynamic threshold as the maximum value of the dynamic factors.
        dynamic_threshold = dynamic_factors.max(-1, keepdim=True).values
        return dynamic_threshold

    def forward(self, attn, normalize=True):
        """mask attention with the right span"""
        # Calculate important scores
        important_scores = self.calculate_important_scores(attn)

        # Calculate dynamic factors and dynamic threshold
        dynamic_factors = self.calculate_dynamic_factors(important_scores)
        dynamic_threshold = self.calculate_dynamic_threshold(dynamic_factors)

        # Apply masking using the dynamic threshold and dynamic factors
        attn = self._mask(attn)
        attn = attn * (important_scores >= dynamic_threshold).unsqueeze(-1).float()

        if normalize:
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1
        return attn

    def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # too fine granularity might be bad for memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            # cache is too short! this happens when validation resumes
            # after a lot of updates.
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe


            
    def get_cache_size(self):
        """determine how long the cache should be"""
        if self._adapt_cache:
            trim_len = self.get_trim_len()
            # give a buffer of 64 steps since a span might increase
            # in future updates
            return min(self._max_span, self._max_span - trim_len + 64)
        else:
            return self._max_span

        
    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_size(self, include_ramp=True):
        return self._mask.get_current_max_size(include_ramp)

    def get_current_avg_size(self, include_ramp=True):
        return self._mask.get_current_avg_size(include_ramp)

    def clamp_param(self):
        self._mask.clamp_param()
