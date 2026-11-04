import argparse
import random
from pathlib import Path
import numpy as np
import torch
from pair_model import batch_tensors
from expert_model import ExpertModel, training_panels, panel_objective
from fusion_core import seed_all, train_head
from training_utils import load_examples, atomic_save, cpu_state, validate_ids

def train_expert(rows, examples, reference_state, kind, seed, epochs, device, path, dropout=None, rank_weight=None, encoder_lr=0.0001, head_lr=0.001):
    if epochs < 1:
        raise ValueError('positive epochs required')
    if any((int(r['target']) not in (0, 1) for r in rows)):
        raise ValueError('binary targets required')
    if dropout is None:
        dropout = 0.1 if kind == 'SharedFT' else 0.0
    if rank_weight is None:
        rank_weight = 0.25 if kind == 'SharedFT' else 1.0
    seed_all(seed)
    constructs = sorted({r['construct_id'] for r in rows})
    model = ExpertModel(kind, reference_state, constructs, dropout).to(device)
    ep = [p for _, p in model.encoder_named_parameters() if p.requires_grad]
    epids = {id(p) for p in ep}
    hp = [p for p in model.parameters() if p.requires_grad and id(p) not in epids]
    optimizer = torch.optim.AdamW([dict(params=ep, lr=encoder_lr), dict(params=hp, lr=head_lr)], weight_decay=0.0001)
    panels = training_panels(rows, range(len(rows)))
    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed * 100003 + epoch)
        order = list(panels)
        rng.shuffle(order)
        total = 0.0
        for panel in order:
            optimizer.zero_grad(set_to_none=True)
            query = list(panel['query'])
            rng.shuffle(query)
            parts = []
            for start in range(0, len(query), 64):
                ii = query[start:start + 64]
                q = model.pair.encode(*batch_tensors([examples[i] for i in ii], device))
                parts.append(model(q, panel['construct']))
            logits = torch.cat(parts)
            y = torch.tensor([int(rows[i]['target']) for i in query], device=device, dtype=torch.float32)
            loss = panel_objective(kind, logits, logits, y, torch.zeros_like(logits), rank_weight) * panel['weight']
            if not torch.isfinite(loss):
                raise ValueError('nonfinite expert loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        print(dict(expert=kind, seed=seed, epoch=epoch, loss=total), flush=True)
    atomic_save(path, dict(model=cpu_state(model), kind=kind, constructs=constructs, calibration=[1.0, 0.0], epoch=epochs, dropout=dropout, rank_weight=rank_weight, encoder_lr=encoder_lr, head_lr=head_lr, seed=seed))
    return model

def train_ranking(path, output, seeds, epochs, device):
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    states = data['states']
    descriptor = data['descriptor']
    support = data['support']
    prior = data['prior']
    panels = data['panels']
    if states.ndim != 2 or states.shape[1] != 128 or descriptor.shape != (len(states), 12):
        raise ValueError('invalid Ranking feature dimensions')
    ids = validate_ids(data['train_ids'], len(states))
    if support.shape != (len(states),) or not np.issubdtype(support.dtype, np.integer) or np.any(support < 0) or np.any(support >= len(states)):
        raise ValueError('invalid support indices')
    if len(data['labels']) != len(states) or len(prior) != len(states) or len(panels) != len(states):
        raise ValueError('row count mismatch')
    labels = {i: int(data['labels'][i]) for i in ids}
    for seed in seeds:
        train_head(states, descriptor, support, prior, panels, labels, ids, seed, epochs, device, Path(output) / f'Ranking_seed_{seed}.pt')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=['Shared', 'Ranking', 'Construct'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--pair')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43, 71, 101])
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dropout', type=float)
    p.add_argument('--rank-weight', type=float)
    p.add_argument('--encoder-lr', type=float, default=0.0001)
    p.add_argument('--head-lr', type=float, default=0.001)
    a = p.parse_args()
    epochs = a.epochs if a.epochs is not None else 120 if a.expert == 'Ranking' else 65
    if epochs < 1:
        p.error('positive epochs required')
    if a.expert == 'Ranking':
        train_ranking(a.data, a.output, a.seeds, epochs, a.device)
        return
    if not a.pair:
        p.error('--pair is required for Shared/Construct')
    rows, examples = load_examples(a.data)
    state = torch.load(a.pair, map_location='cpu', weights_only=True)['model']
    kind = {'Shared': 'SharedFT', 'Construct': 'ConstructFT'}[a.expert]
    for seed in a.seeds:
        train_expert(rows, examples, state, kind, seed, epochs, a.device, Path(a.output) / f'{a.expert}_seed_{seed}.pt', a.dropout, a.rank_weight, a.encoder_lr, a.head_lr)
if __name__ == '__main__':
    main()
