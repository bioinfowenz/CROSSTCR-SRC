import math
import torch
from torch import nn
from torch.nn import functional as F

def microbatch_loss(logits, targets, weights, n_train, effective_count):
    if not 0 < len(logits) <= effective_count <= n_train:
        raise ValueError('invalid batch counts')
    if logits.shape != targets.shape or targets.shape != weights.shape:
        raise ValueError('shape mismatch')
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all() or (not (weights > 0).all()):
        raise ValueError('nonfinite or nonpositive training values')
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError('nonbinary auxiliary targets')
    return n_train / effective_count * (weights * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')).sum()

def batch_tensors(examples, device='cpu'):
    lengths = [sum((len(x) for x in e)) for e in examples]
    if not lengths or any((len(e) != 4 or any((x.ndim != 2 or x.shape[1] != 640 or (not len(x)) for x in e)) for e in examples)):
        raise ValueError('four nonempty residue segments required')
    b, l = (len(examples), max(lengths))
    features = torch.zeros(b, l, 640)
    segments = torch.full((b, l), -1, dtype=torch.long)
    positions = torch.zeros(b, l, dtype=torch.long)
    for i, e in enumerate(examples):
        offset = 0
        for s, x in enumerate(e):
            n = len(x)
            features[i, offset:offset + n] = x
            segments[i, offset:offset + n] = s
            positions[i, offset:offset + n] = torch.arange(n)
            offset += n
    return tuple((t.to(device) for t in (features, segments, positions)))

class AuxiliaryModel(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(640, 128)
        self.segment_embedding = nn.Embedding(4, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 512, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pool_projection = nn.Sequential(nn.Linear(4 * 128, 128), nn.LayerNorm(128))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, features, segments, positions):
        padding = segments < 0
        angles = positions[..., None].to(features.dtype) * torch.exp(torch.arange(0, 128, 2, device=features.device) * (-math.log(10000.0) / 128))
        pos = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        h = self.projection(features) + self.segment_embedding(segments.clamp_min(0)) + pos
        h = self.encoder(h, src_key_padding_mask=padding)
        means = []
        for s in range(4):
            mask = (segments == s).unsqueeze(-1)
            if (mask.sum(dim=1) == 0).any():
                raise ValueError('missing model segment')
            means.append((h * mask).sum(dim=1) / mask.sum(dim=1))
        return self.head(self.pool_projection(torch.cat(means, dim=-1))).squeeze(-1)
