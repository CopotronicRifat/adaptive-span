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

        # dynamic factor and threshold
        self.dynamic_factor = nn.Parameter(torch.ones(1))
        self.dynamic_threshold = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, target):
        """
        Input:
            - `hidden` FloatTensor(shape + (d_proj,))
            - `target` LongTensor(shape)
        Output:
            - `nll` FloatTensor(shape)
        """
        assert hidden.shape[-1] == self.d_proj
        assert hidden.shape[:-1] == target.shape
        shape = target.shape
        hidden = hidden.view(-1, self.d_proj)
        target = target.view(-1)

        # calculate dynamic mask
        important_score = F.relu(hidden * self.dynamic_factor)
        important_mask = important_score > self.dynamic_threshold

        # construct weights and biases
        weights, biases = [], []
        for i in range(len(self.cutoffs)):
            weight_i = self.out_layers[i].weight
            bias_i = self.out_layers[i].bias
            if i == 0:
                weight_i = torch.cat([weight_i, self.cluster_proj.weight], dim=0)
                bias_i = torch.cat([bias_i, self.cluster_proj.bias], dim=0)
            weights.append(weight_i)
            biases.append(bias_i)

        # head / cluster assignments
        head_logit = self._compute_logit(hidden, weights[0], biases[0], self.out_projs[0])
        head_logprob = F.log_softmax(head_logit.float(), dim=1)

        # final log-probabilities
        nll = torch.zeros_like(target, dtype=torch.float32, device=hidden.device)

        offset = 0
        cutoff_values = [0] + self.cutoffs

        # for each cluster
        for i in range(len(cutoff_values) - 1):

            # select the target tokens in that cluster
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
            mask_i = (target >= l_idx) & (target < r_idx) & important_mask
            indices_i = mask_i.non



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

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()

