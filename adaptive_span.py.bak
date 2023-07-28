import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMask(nn.Module):
    """ An adaptive mask module from "Adaptive Input Representations for
    Neural Language Modeling" (https://arxiv.org/abs/1809.10853) """
    def __init__(self, n_tokens, d_embed, d_proj, cutoffs, div_val=4):
        super(AdaptiveMask, self).__init__()

        self.n_tokens = n_tokens
        self.d_embed = d_embed
        self.d_proj = d_proj

        assert 0 < min(cutoffs) <= max(cutoffs) < n_tokens
        self.cutoffs = cutoffs + [n_tokens]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        assert self.div_val > 1
        assert len(self.cutoffs) > 1

        self.emb_scale = d_proj ** 0.5

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        # embedding layers / projections
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // (div_val ** i)
            self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
            self.emb_projs.append(nn.Linear(d_emb_i, d_proj).weight)

        # dynamic factor & threshold
        self.important_score = nn.Linear(d_embed, 1)
        self.dynamic_factor = nn.Parameter(torch.ones(1), requires_grad=True)
        self.threshold = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, indices):
        param = self.emb_layers[0].weight.data
        idx_flat = indices.contiguous().view(-1)
        emb_flat = torch.zeros([idx_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)

        # for each cluster
        for i in range(len(self.cutoffs)):
            # find elements in that cluster
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            mask_i = (idx_flat >= l_idx) & (idx_flat < r_idx)

            # if there are no elements, continue
            indices_i = mask_i.nonzero().squeeze()
            if indices_i.numel() == 0:
                continue

            # add embeddings from this cluster
            idx_i = idx_flat.index_select(0, indices_i) - l_idx
            emb_i = self.emb_layers[i](idx_i)
            emb_i = F.linear(emb_i, self.emb_projs[i])
            emb_flat = emb_flat.type_as(emb_i) if emb_flat.dtype != emb_i.dtype else emb_flat  # small hack for AMP-O1
            emb_flat.index_copy_(0, indices_i, emb_i)

        # reshape embeddings
        embed = emb_flat.view(*indices.size(), self.d_proj)

        # rescale embeddings
        embed.mul_(self.emb_scale)

        # calculate important scores
        important_scores = self.important_score(embed).squeeze(-1)

        # calculate dynamic mask
        dynamic_mask = important_scores > self.dynamic_factor * self.threshold

        return dynamic_mask
    
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
    def __init__(self, input_size, hidden_size, attn_span, nb_heads, **kwargs):
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

