# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



from transformers import BertModel, BertTokenizer

class AdaptiveMask(nn.Module):
    def __init__(self, max_size, ramp_size, init_val, shape):
        super(AdaptiveMask, self).__init__()
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.register_buffer("mask_template", torch.arange(0, self._max_size).float())
        self.register_buffer("current_val", torch.tensor(init_val).float())


        # Assuming you have self.Wa initialized for attention scoring

    def get_token_embedding(self, token_indices):
        """
        Function to get contextualized embeddings for tokens in a sentence using BERT.
        
        Parameters:
        token_indices (torch.Tensor): A tensor containing indices of tokens in a sentence.
        
        Returns:
        torch.Tensor: Contextualized embeddings for tokens in the sentence.
        """
        # Assuming token_indices is a tensor of shape (max_sentence_length,)
        # Convert token indices to words (using a vocabulary or any other mapping)
        tokens = [self.vocab[token_idx] for token_idx in token_indices]

        # Convert tokens to BERT input format
        inputs = self.tokenizer(" ".join(tokens), return_tensors="pt", padding=True, truncation=True)

        # Get BERT embeddings for tokens
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Get the last layer hidden states (contextualized embeddings)
        token_embeddings = outputs.last_hidden_state

        return token_embeddings

    def calculate_important_scores(self, x):
        # Assuming x is a tensor representing a batch of sentences, with shape (batch_size, max_sentence_length)

        # Tokenization (split sentences into individual tokens)
        tokens_list = [sentence.split() for sentence in x]

        # Convert tokens into token indices using a vocabulary (for demonstration, we create random token indices)
        # In a real scenario, you would use a vocabulary and convert words to their corresponding indices.
        vocab_size = len(self.word_embeddings.weight)
        token_indices_list = [[torch.randint(vocab_size, size=(self.max_sentence_length,))] for tokens in tokens_list]

        # Embedding (using BERT)
        embeddings_list = [self.get_token_embedding(token_indices) for token_indices in token_indices_list]

        # Attention Scoring
        important_scores_list = []
        for embeddings in embeddings_list:
            # Apply a linear transformation to embeddings
            linear_transform = torch.matmul(embeddings, self.Wa)

            # Apply softmax to get attention weights
            attention_weights = F.softmax(linear_transform, dim=-1)

            # Calculate the important scores by taking a weighted sum of embeddings using attention weights
            important_scores = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=-2)

            # Append important scores to the list
            important_scores_list.append(important_scores)

        # Stack the important scores back into a tensor
        important_scores = torch.stack(important_scores_list, dim=0)

        return important_scores

    def calculate_dynamic_factors(self, important_scores):
        # Apply linear transformation to the important scores to get the dynamic factors
        dynamic_factors = self.linear(important_scores)
        return dynamic_factors

    def calculate_dynamic_threshold(self, dynamic_factors):
        # Calculate dynamic threshold here (You need to implement this function)
        # For example, you could apply some aggregation operation to dynamic factors
        dynamic_threshold = torch.mean(dynamic_factors)
        return dynamic_threshold

    def forward(self, x):
        important_scores = self.calculate_important_scores(x)
        dynamic_factors = self.calculate_dynamic_factors(important_scores)
        dynamic_threshold = self.calculate_dynamic_threshold(dynamic_factors)

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



class AdaptiveSpan(nn.Module):
    def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
                 adapt_span_init, adapt_span_cache, nb_heads):
        nn.Module.__init__(self)
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                 ramp_size=adapt_span_ramp,
                                 init_val=adapt_span_init,
                                 shape=(nb_heads, 1, 1))

    def forward(self, attn, normalize=True):
        """mask attention with the right span"""
        # batch and head dimensions are merged together, so separate them first
        B = attn.size(0) # batch size
        M = attn.size(1) # block size
        attn = attn.reshape(B // self._nb_heads, self._nb_heads, M, -1)

        attn = self._mask(attn)
        if normalize:
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1

        attn = attn.view(B, M, -1)
        return attn

    def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # too fine granularity might be bad for the memory management
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

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()
