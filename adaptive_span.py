import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel


class PreTrainedBERTEmbedding(nn.Module):
    def __init__(self, pretrained_model_name):
        super(PreTrainedBERTEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, tokens):
        input_ids = self.tokenizer(tokens, padding=True, truncation=True, return_tensors='pt')['input_ids']
        embeddings = self.bert_model(input_ids)[0]
        return embeddings


class AdaptiveMask(nn.Module):
    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,), embedding_model=None):
        super(AdaptiveMask, self).__init__()
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)
        self.embedding_model = embedding_model  # Pass the embedding model instance to the class

    def calculate_important_scores(self, x):
        # Assuming x is a tensor representing a batch of sentences, with shape (batch_size, max_sentence_length)

        # Tokenization (split sentences into individual tokens)
        # We use BERT tokenizer instead of the simple split() function
        tokens_list = [self.embedding_model.tokenizer.tokenize(sentence) for sentence in x]

        embeddings_list = [self.embedding_model(tokens) for tokens in tokens_list]

        # Attention Scoring
        attention_scores_list = []
        for embeddings in embeddings_list:
            # Apply softmax to get attention weights
            attention_weights = F.softmax(embeddings, dim=-1)

            # Append attention scores to the list
            attention_scores_list.append(attention_weights)

        # Stack the attention scores back into a tensor
        attention_scores = torch.stack(attention_scores_list, dim=0)

        return attention_scores

    def get_token_index(self, token):
        # A simple function to convert token to its index in the vocabulary
        # For demonstration purposes, we'll assume a small vocabulary with 100 tokens
        # In a real scenario, you would have a proper vocabulary and token-to-index mapping.
        vocab = ["token" + str(i) for i in range(100)]
        return vocab.index(token)


    def calculate_dynamic_factors(self, important_scores):
        # Calculate dynamic factors here (You need to implement this function)
        # For example, you could apply some transformations to important scores
        dynamic_factors = important_scores * 2.0  # Placeholder implementation
        return dynamic_factors

    def calculate_dynamic_threshold(self, dynamic_factors):
        # Calculate dynamic threshold here (You need to implement this function)
        # For example, you could apply some aggregation operation to dynamic factors
        dynamic_threshold = torch.mean(dynamic_factors)  # Placeholder implementation
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