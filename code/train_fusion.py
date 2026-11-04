import argparse
from pathlib import Path
import numpy as np
import torch
from fusion_core import rank_pool, train_gate
from training_utils import validate_ids, atomic_save

def fit(data_path, output, setting='familiar', seeds=(17, 29, 43, 71, 101), epochs=120, device='cpu'):
    if setting not in ('familiar', 'tcr-cold', 'pmhc-cold', 'double-cold'):
        raise ValueError('unknown training setting')
    if epochs < 1:
        raise ValueError('positive epochs required')
    with np.load(data_path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    panels = data['panels']
    ids = validate_ids(data['train_ids'], len(panels))
    scores = data['seed_scores']
    context = data['context']
    names = data['expert_order'].tolist()
    if set(names) != {'Shared', 'Ranking', 'Construct'} or len(names) != 3:
        raise ValueError('three distinct SRC experts required')
    if scores.shape != (len(ids), 3, 5) or context.shape != (len(ids), 6):
        raise ValueError('invalid OOF gate arrays')
    if not np.isfinite(scores).all() or not np.isfinite(context).all():
        raise ValueError('nonfinite gate input')
    if 'query_ids' in data and set(ids) & set(data['query_ids'].tolist()):
        raise ValueError('query included in gate training')
    if len(data['labels']) != len(panels):
        raise ValueError('label row count mismatch')
    labels = {i: int(data['labels'][i]) for i in ids}
    if any((v not in (0, 1) for v in labels.values())):
        raise ValueError('binary training labels required')
    mean_scores = data['mean_scores']
    if mean_scores.shape != (len(ids), 3) or not np.isfinite(mean_scores).all():
        raise ValueError('invalid ensemble means')
    means, std = rank_pool(scores, panels[ids], mean_scores)
    for seed in seeds:
        path = Path(output) / f'fusion_seed_{seed}.pt'
        model, scaler, receipt = train_gate(means, std, context, panels, labels, ids, seed, epochs, device, path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        saved['expert_order'] = names
        saved['setting'] = setting
        atomic_save(path, saved)
    return (means, std)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--setting', choices=['familiar', 'tcr-cold', 'pmhc-cold', 'double-cold'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43, 71, 101])
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    fit(a.data, a.output, a.setting, a.seeds, a.epochs, a.device)
if __name__ == '__main__':
    main()
