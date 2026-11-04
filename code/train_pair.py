import argparse
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from pair_model import AuxiliaryModel, batch_tensors, microbatch_loss
from fusion_core import seed_all
from training_utils import load_examples, atomic_save, cpu_state, binary_metrics

def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos = logits[targets == 1]
    neg = logits[targets == 0]
    if not len(pos) or not len(neg):
        return logits.sum() * 0
    return F.softplus(-(pos[:, None] - neg[None, :])).mean()

def better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return (candidate['validation_AUROC'], candidate['validation_AP'], -candidate['epoch']) > (best['validation_AUROC'], best['validation_AP'], -best['epoch'])

def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, rows: list[dict], examples: list, order: list[int], device: str) -> dict:
    model.train()
    total = 0.0
    for start in range(0, len(order), 128):
        ids = order[start:start + 128]
        optimizer.zero_grad(set_to_none=True)
        pieces = []
        for off in range(0, len(ids), 16):
            subset = ids[off:off + 16]
            pieces.append(model(*batch_tensors([examples[i] for i in subset], device)))
        logits = torch.cat(pieces)
        targets = torch.tensor([float(rows[i]['auxiliary_target']) for i in ids], device=device)
        weights = torch.tensor([float(rows[i]['training_weight']) for i in ids], device=device)
        bce = microbatch_loss(logits, targets, weights, len(rows), len(ids))
        rank = pairwise_rank_loss(logits, targets)
        loss = 0.75 * bce + 0.25 * rank
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not math.isfinite(float(loss.detach().cpu())):
            raise ValueError('nonfinite rank training loss')
        total += float(loss.detach().cpu()) * len(ids)
    return {'training_loss_estimate': total / len(order)}

def fit(train_path, validation_path, output, seed=42, device='cpu', epochs=100, patience=10):
    rows, examples = load_examples(train_path)
    validation, validation_examples = load_examples(validation_path)
    for collection in (rows, validation):
        if any((int(r['auxiliary_target']) not in (0, 1) for r in collection)):
            raise ValueError('binary auxiliary targets required')
    if {int(r['auxiliary_target']) for r in validation} != {0, 1}:
        raise ValueError('validation requires both classes')
    if epochs < 1 or patience < 1:
        raise ValueError('positive epochs/patience required')
    seed_all(seed)
    model = AuxiliaryModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    output = Path(output)
    best = None
    bad = 0
    for epoch in range(1, epochs + 1):
        result = train_epoch(model, optimizer, rows, examples, torch.randperm(len(rows)).tolist(), device)
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(validation_examples), 16):
                scores.extend(model(*batch_tensors(validation_examples[start:start + 16], device)).sigmoid().cpu().tolist())
        metrics = binary_metrics([int(r['auxiliary_target']) for r in validation], scores)
        candidate = dict(epoch=epoch, validation_AUROC=metrics['AUROC'], validation_AP=metrics['AP'])
        selected = better(candidate, best)
        bad = 0 if selected else bad + 1
        if selected:
            best = candidate
            atomic_save(output / 'best.pt', dict(model=cpu_state(model), seed=seed, selection=best))
        atomic_save(output / 'last.pt', dict(model=cpu_state(model), seed=seed, epoch=epoch, selection=best))
        print(dict(**candidate, **result), flush=True)
        if bad >= patience:
            break
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 2026, 20260713, 31415, 27182])
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    a = p.parse_args()
    for seed in a.seeds:
        fit(a.train, a.validation, Path(a.output) / f'seed_{seed}', seed, a.device, a.epochs, a.patience)
if __name__ == '__main__':
    main()
