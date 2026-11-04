from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def average_ranks(scores):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or (not np.isfinite(x).all()):
        raise ValueError('Invalid ranking scores')
    order = np.argsort(x, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(x[order]) != 0) + 1, len(x)]
    out = np.empty(len(x), float)
    for a, b in zip(cuts[:-1], cuts[1:]):
        out[order[a:b]] = (a + b) / (2 * len(x))
    return out

def bounded_tiebreak(prior, residual):
    p = np.asarray(prior, dtype=float)
    r = np.asarray(residual, dtype=float)
    if p.ndim != 1 or p.shape != r.shape or (not len(p)) or (not np.isfinite(p + r).all()):
        raise ValueError('Invalid prior/correction')
    gaps = np.diff(np.unique(p))
    scale = 0.49 * float(gaps.min()) if len(gaps) else 1.0
    return p + scale * np.tanh(r)

def fit_scaler(x, ids, floor=0.1):
    ids = list(ids)
    x = np.asarray(x, dtype=float)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Invalid scaler membership')
    return dict(mean=x[ids].mean(0), std=np.maximum(x[ids].std(0), floor))

def assert_access(rows, train, query, support):
    train = list(train)
    query = list(query)
    support = list(support)
    if any((len(v) != len(set(v)) for v in (train, query, support))):
        raise ValueError('Duplicate access')
    if set(train) & set(query) or set(query) & set(support):
        raise ValueError('Query used in fit/support')
    if {rows[i]['key'] for i in query} & {rows[i]['key'] for i in train + support}:
        raise ValueError('Target alias in fit/support')
    if {rows[i]['component'] for i in train} & {rows[i]['component'] for i in query}:
        raise ValueError('Held-out component in training')
    return True

def rank_pool(seed_scores, panels, mean_scores=None):
    s = np.asarray(seed_scores, float)
    panels = np.asarray(panels)
    if s.ndim != 3 or len(s) != len(panels):
        raise ValueError('Expected rows x experts x seeds')
    source = s.mean(2) if mean_scores is None else np.asarray(mean_scores, float)
    means = np.zeros(s.shape[:2])
    ranks = np.zeros_like(s)
    for panel in np.unique(panels):
        ii = np.flatnonzero(panels == panel)
        for e in range(s.shape[1]):
            means[ii, e] = average_ranks(source[ii, e])
            for seed in range(s.shape[2]):
                ranks[ii, e, seed] = average_ranks(s[ii, e, seed])
    return (means, ranks.std(2))

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TransferHead(nn.Module):

    def __init__(self, qscale, dscale, seed=17):
        super().__init__()
        seed_all(seed)
        for name, values in [('qm', qscale['mean']), ('qs', qscale['std']), ('dm', dscale['mean']), ('ds', dscale['std'])]:
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32))
        self.project = nn.Linear(128, 16, bias=False)
        self.hidden = nn.Linear(60, 16)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, q, s, d):
        q = torch.tanh(self.project((q - self.qm) / self.qs))
        s = torch.tanh(self.project((s - self.qm) / self.qs))
        x = torch.cat([q - s, torch.abs(q - s), q * s, (d - self.dm) / self.ds], 1)
        return self.output(self.dropout(torch.tanh(self.hidden(x)))).squeeze(1)

def ranking_pairs(ids, panels, labels, prior=None):
    ids = np.asarray(ids, int)
    panels = np.asarray(panels)
    pos = []
    neg = []
    weight = []
    details = []
    for panel in sorted(set(panels[ids])):
        local = np.flatnonzero(panels[ids] == panel)
        groups = [local] if prior is None else [local[np.asarray(prior)[ids[local]] == v] for v in np.unique(np.asarray(prior)[ids[local]])]
        records = []
        for g in groups:
            pp = [j for j in g if int(labels[int(ids[j])]) == 1]
            nn_ = [j for j in g if int(labels[int(ids[j])]) == 0]
            if not pp or not nn_:
                continue
            discount = 1.0 if prior is None else 1 / np.log2(2 + np.sum(np.asarray(prior)[ids[local]] > np.asarray(prior)[ids[g[0]]]) + (len(g) - 1) / 2)
            records.append((pp, nn_, discount))
        if not records:
            continue
        total = sum((v[2] for v in records))
        for pp, nn_, discount in records:
            pos.extend(np.repeat(pp, len(nn_)).tolist())
            neg.extend(np.tile(nn_, len(pp)).tolist())
            weight.extend([discount / (total * len(pp) * len(nn_))] * (len(pp) * len(nn_)))
        details.append(dict(panel_id=str(panel), informative_groups=len(records), pairs=sum((len(p) * len(n) for p, n, _ in records))))
    if not details:
        raise ValueError('No informative measured positive/negative ranking groups')
    return (np.asarray(pos, int), np.asarray(neg, int), np.asarray(weight, float) / len(details), details)

def train_head(states, descriptor, support, prior, panels, labels, ids, seed, epochs, device, path):
    ids = list(ids)
    if set(ids) - set(labels):
        raise ValueError('Training label missing')
    if not all((int(labels[i]) in (0, 1) for i in ids)):
        raise ValueError('Only measured binary labels are permitted')
    qscale = fit_scaler(states, ids)
    dscale = fit_scaler(descriptor, ids, 0.25)
    model = TransferHead(qscale, dscale, seed).to(device)
    initial = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    q = torch.as_tensor(states[ids], dtype=torch.float32, device=device)
    s = torch.as_tensor(states[np.asarray(support)[ids]], dtype=torch.float32, device=device)
    d = torch.as_tensor(descriptor[ids], dtype=torch.float32, device=device)
    p, n, w, groups = ranking_pairs(ids, panels, labels, prior)
    p = torch.as_tensor(p, device=device)
    n = torch.as_tensor(n, device=device)
    w = torch.as_tensor(w, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(q, s, d)
        loss = (F.softplus(-(z[p] - z[n]) / 0.5) * w).sum() + 0.001 * z.square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite transfer loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(float(loss.detach()))
    model.eval()
    final = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    receipt = dict(seed=seed, epochs=epochs, optimizer_steps=epochs, train_ids=ids, parameter_count=sum((p.numel() for p in model.parameters())), parameter_change_l2=float(torch.linalg.vector_norm(final - initial)), loss_first=history[0], loss_last=history[-1], loss_history=history, ranking_groups=groups, target_labels_for_selection=False, encoder_updated=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, receipt=receipt), path)
    return (model, receipt)

def load_head(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['model']
    model = TransferHead(dict(mean=state['qm'], std=state['qs']), dict(mean=state['dm'], std=state['ds']))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_head(model, states, descriptor, support, ids, device):
    ids = list(ids)
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(states[ids], dtype=torch.float32, device=device), torch.as_tensor(states[np.asarray(support)[ids]], dtype=torch.float32, device=device), torch.as_tensor(descriptor[ids], dtype=torch.float32, device=device)).cpu().numpy().astype(float)

class RankGate(nn.Module):

    def __init__(self, n_experts, seed=17):
        super().__init__()
        seed_all(seed)
        self.n_experts = n_experts
        self.hidden = nn.Linear(2 * n_experts + 6, 16)
        self.output = nn.Linear(16, n_experts)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, p):
        w = torch.softmax(self.output(torch.tanh(self.hidden(x))), 1)
        return ((w * p).sum(1), w)

def train_gate(means, std, context, panels, labels, ids, seed, epochs, device, path):
    raw = np.concatenate([means, std, context], 1)
    scaler = fit_scaler(raw, range(len(raw)), 1e-06)
    x = torch.as_tensor((raw - scaler['mean']) / scaler['std'], dtype=torch.float32, device=device)
    prob = torch.as_tensor(means, dtype=torch.float32, device=device)
    model = RankGate(means.shape[1], seed).to(device)
    p, n, w, details = ranking_pairs(ids, panels, labels)
    p = torch.as_tensor(p, device=device)
    n = torch.as_tensor(n, device=device)
    w = torch.as_tensor(w, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        score, _ = model(x, prob)
        loss = (F.softplus(-(score[p] - score[n]) / 0.1) * w).sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite gate loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(float(loss.detach()))
    receipt = dict(seed=seed, epochs=epochs, optimizer_steps=epochs, train_ids=list(ids), loss_first=history[0], loss_last=history[-1], loss_history=history, n_experts=means.shape[1], parameter_count=sum((p.numel() for p in model.parameters())), ranking_groups=details, target_labels_for_selection=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, scaler={k: torch.as_tensor(v, dtype=torch.float64) for k, v in scaler.items()}, receipt=receipt), path)
    return (model, scaler, receipt)

def predict_gate(model, scaler, means, std, context, device):
    raw = np.concatenate([means, std, context], 1)
    model.eval()
    with torch.no_grad():
        pred, w = model(torch.as_tensor((raw - scaler['mean']) / scaler['std'], dtype=torch.float32, device=device), torch.as_tensor(means, dtype=torch.float32, device=device))
    return (pred.cpu().numpy().astype(float), w.cpu().numpy().astype(float))

def load_gate(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = RankGate(saved['receipt']['n_experts'])
    model.load_state_dict(saved['model'])
    model.to(device)
    model.eval()
    return (model, {k: v.numpy() for k, v in saved['scaler'].items()})
