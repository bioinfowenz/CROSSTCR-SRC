import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from pair_model import batch_tensors

def binary_metrics(labels, scores):
    if len(labels) != len(scores) or not labels or set(labels) - {0, 1}:
        raise ValueError('invalid metric input')
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return {'AUROC': None, 'AP': None, 'reason': 'single_class'}
    buckets = defaultdict(lambda: [0, 0])
    for y, s in zip(labels, scores):
        if not math.isfinite(s):
            raise ValueError('nonfinite score')
        buckets[float(s)][0 if y else 1] += 1
    tp = fp = auc = ap = 0.0
    for score in sorted(buckets, reverse=True):
        p, n = buckets[score]
        tp += p
        fp += n
        auc += p * (neg - fp + n / 2)
        ap += p * tp / (tp + fp)
    return {'AUROC': auc / (pos * neg), 'AP': ap / pos, 'reason': ''}

def load_examples(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    rows = data['rows']
    examples = data['examples']
    if not rows or len(rows) != len(examples):
        raise ValueError('rows/examples length mismatch')
    for row, e in zip(rows, examples):
        batch_tensors([e])
        if any((not torch.isfinite(t).all() for t in e)):
            raise ValueError('nonfinite residue features')
    return (rows, examples)

def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.pt')
    torch.save(payload, tmp)
    tmp.replace(path)

def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

def validate_ids(ids, n):
    raw = np.asarray(ids)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('integer IDs required')
    values = raw.tolist()
    if not values or len(set(values)) != len(values) or min(values) < 0 or (max(values) >= n):
        raise ValueError('invalid training IDs')
    return values
