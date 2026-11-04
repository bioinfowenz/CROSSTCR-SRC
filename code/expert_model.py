import math
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

class ExplicitEncoderLayer(nn.TransformerEncoderLayer):

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_mask is not None or is_causal:
            raise ValueError('R1 uses noncausal unmasked pair attention')
        x = src
        z = self.norm1(x)
        attn = self.self_attn
        q, k, v = F.linear(z, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        batch, width, dim = q.shape
        heads = attn.num_heads
        split = lambda t: t.reshape(batch, width, heads, dim // heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(dim // heads)
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(pad, -math.inf) if pad.dtype == torch.bool else scores + pad
        weights = F.dropout(scores.softmax(-1), p=attn.dropout, training=self.training)
        values = (weights @ v).transpose(1, 2).reshape(batch, width, dim)
        x = x + self.dropout1(F.linear(values, attn.out_proj.weight, attn.out_proj.bias))
        z = self.norm2(x)
        return x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(z)))))

class PairEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(640, 128)
        self.segment_embedding = nn.Embedding(4, 128)
        layer = ExplicitEncoderLayer(128, 4, 512, 0.1, activation=lambda x: F.gelu(x, approximate='tanh'), batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pool_projection = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(approximate='tanh'), nn.Dropout(0.1), nn.Linear(64, 1))

    def encode(self, features, segments, positions):
        features = features.detach()
        a = positions[..., None].to(features.dtype) * torch.exp(torch.arange(0, 128, 2, device=features.device) * (-math.log(10000.0) / 128))
        pos = torch.stack((a.sin(), a.cos()), -1).flatten(-2)
        z = self.projection(features) + self.segment_embedding(segments.clamp_min(0)) + pos
        z = self.encoder(z, src_key_padding_mask=segments < 0)
        means = []
        for channel in range(4):
            mask = (segments == channel).unsqueeze(-1)
            if (mask.sum(1) == 0).any():
                raise ValueError('missing R1 channel')
            means.append((z * mask).sum(1) / mask.sum(1))
        return self.pool_projection(torch.cat(means, -1))

class PairHead(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 64), nn.GELU(approximate='tanh'), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, q):
        return self.net(q).squeeze(-1)

class QueryResidual(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 32), nn.GELU(approximate='tanh'), nn.Dropout(dropout), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q):
        return self.net(q).squeeze(-1)

def construct(row):
    return row['construct_id']

def training_panels(rows, ids):
    groups = defaultdict(list)
    for i in ids:
        groups[rows[i]['panel_id']].append(i)
    components = defaultdict(set)
    for panel, ii in groups.items():
        if len({(rows[i]['r01_component_id'], construct(rows[i])) for i in ii}) != 1:
            raise ValueError('panel spans components or constructs')
        components[rows[ii[0]]['r01_component_id']].add(panel)
    return [dict(panel=p, component=rows[ii[0]]['r01_component_id'], construct=construct(rows[ii[0]]), query=ii, weight=1 / (len(components) * len(components[rows[ii[0]]['r01_component_id']]))) for p, ii in sorted(groups.items())]

def panel_objective(kind, logits, secondary, y, sim, rank_weight):
    positive = y == 1
    negative = y == 0
    if kind == 'TieFT':
        rank = secondary.sum() * 0
        if positive.any() and negative.any():
            ties = sim[positive][:, None] == sim[negative][None]
            pair = F.softplus(secondary[negative][None] - secondary[positive][:, None])
            if ties.any():
                rank = pair[ties].mean()
        return rank + 0.01 * secondary.square().mean()
    rank = logits.sum() * 0
    if positive.any() and negative.any():
        rank = F.softplus(logits[negative][None] - logits[positive][:, None]).mean()
    return F.binary_cross_entropy_with_logits(logits, y) + rank_weight * rank

class ExpertModel(nn.Module):

    def __init__(self, kind, reference_state, constructs, dropout):
        super().__init__()
        if kind not in ('SharedFT', 'ConstructFT'):
            raise ValueError('unknown expert')
        self.kind = kind
        self.constructs = tuple(sorted(constructs))
        self.mapping = {c: str(i) for i, c in enumerate(self.constructs)}
        self.pair = PairEncoder()
        self.pair.load_state_dict(reference_state, strict=True)
        self.pair.requires_grad_(True)
        self.pair.head.requires_grad_(False)
        self.head = PairHead(dropout)
        if kind == 'ConstructFT':
            self.residuals = nn.ModuleDict({self.mapping[c]: QueryResidual(dropout) for c in self.constructs})

    def train(self, mode=True):
        super().train(mode)
        self.pair.head.eval()
        return self

    def forward(self, q, construct):
        score = self.head(q)
        if self.kind == 'ConstructFT' and construct in self.mapping:
            score = score + self.residuals[self.mapping[construct]](q)
        return score

    def encoder_named_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith('pair.') and '.head.' not in n:
                yield (n, p)
