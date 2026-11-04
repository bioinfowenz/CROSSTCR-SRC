import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def rank_intervals(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('finite nonempty score vector required')
    return rankdata(-scores, method='min').astype(int), rankdata(-scores, method='max').astype(int)


def validate_ranks(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if lo.shape != hi.shape or lo.ndim != 1 or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError('invalid rank intervals')
    if np.any(lo < 1) or np.any(hi < lo) or np.any(lo != np.floor(lo)) or np.any(hi != np.floor(hi)):
        raise ValueError('positive integer rank intervals required')
    return lo, hi


def recall_at_k(lo, hi, positive, k):
    lo, hi = validate_ranks(lo, hi)
    positive = np.asarray(positive, dtype=bool)
    if positive.shape != lo.shape or not positive.any() or int(k) != k or k < 0:
        raise ValueError('positive membership and nonnegative integer budget required')
    probability = np.clip(k - lo[positive] + 1, 0, hi[positive] - lo[positive] + 1) / (hi[positive] - lo[positive] + 1)
    return float(probability.mean())


def compare_rankings(reference, alternative, k=10, key='peptide'):
    if k < 1 or int(k) != k:
        raise ValueError('positive integer top-k required')
    if reference[key].duplicated().any() or alternative[key].duplicated().any():
        raise ValueError('duplicate candidate identity')
    if set(reference[key]) != set(alternative[key]) or reference.empty:
        raise ValueError('matching nonempty candidate sets required')
    a = reference.set_index(key)
    b = alternative.set_index(key).loc[a.index]
    al, ah = validate_ranks(a['rank_min'], a['rank_max'])
    bl, bh = validate_ranks(b['rank_min'], b['rank_max'])
    ta = set(a.index[al <= k])
    tb = set(b.index[bl <= k])
    union = ta | tb
    rho = float(spearmanr((al + ah) / 2, (bl + bh) / 2).statistic)
    return dict(n=len(a), top_k=k, intersection=len(ta & tb), union=len(union), jaccard=len(ta & tb) / len(union) if union else np.nan, spearman=rho)


def analyze(scores, output, positives=None, budgets=None, group_by=None, score_column='score', key='peptide', positive_column='label_screen', variant_columns=(), comparison=None, top_k=10):
    frame = pd.read_csv(scores, sep='\t')
    if key not in frame:
        raise ValueError('candidate identity column missing')
    if frame[key].isna().any():
        raise ValueError('missing candidate identity')
    if group_by is None:
        group_by = [c for c in ('case_id', 'pool_name', 'pool', 'method', 'support_setting') if c in frame]
    if any(c not in frame for c in group_by):
        raise ValueError('grouping column missing')
    positive_set = None
    if positives:
        table = pd.read_csv(positives, sep='\t')
        if positive_column in table:
            table = table.loc[table[positive_column] == 1]
        positive_set = set(table[key].dropna())
        if not positive_set:
            raise ValueError('no supplied positives')
    compared = pd.read_csv(comparison, sep='\t') if comparison else None
    groups = frame.groupby(group_by, dropna=False, sort=False) if group_by else [((), frame)]
    ranked, recalls, stability = [], [], []
    for values, sub in groups:
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(group_by, values))
        sub = sub.copy()
        if sub[key].duplicated().any():
            raise ValueError('duplicate candidate within group')
        if not {'rank_min', 'rank_max'}.issubset(sub.columns):
            sub['rank_min'], sub['rank_max'] = rank_intervals(sub[score_column])
        lo, hi = validate_ranks(sub['rank_min'], sub['rank_max'])
        sub['rank_min'], sub['rank_max'] = lo.astype(int), hi.astype(int)
        ranked.append(sub.sort_values(['rank_min', 'rank_max', key], kind='stable'))
        if positive_set is not None:
            positive = sub[key].isin(positive_set).to_numpy()
            if not positive_set.issubset(set(sub[key])):
                raise ValueError('positive candidate missing from scored pool')
        elif positive_column in sub:
            positive = (sub[positive_column] == 1).to_numpy()
        else:
            positive = np.zeros(len(sub), dtype=bool)
        if positive.any():
            ks = list(budgets) if budgets is not None else range(int(hi.max()) + 1)
            for k in ks:
                r = recall_at_k(lo, hi, positive, k)
                recalls.append(dict(**identity, K=k, recall=r, expected_positives_found=r * int(positive.sum()), n_positives=int(positive.sum())))
        for column in variant_columns:
            other = sub[[key]].copy()
            other['rank_min'], other['rank_max'] = rank_intervals(sub[column])
            stability.append(dict(**identity, variant=column, **compare_rankings(sub, other, top_k, key)))
        if compared is not None:
            other = compared
            for c, v in identity.items():
                other = other.loc[other[c].isna() if pd.isna(v) else other[c] == v]
            other = other.copy()
            if not {'rank_min', 'rank_max'}.issubset(other.columns):
                other['rank_min'], other['rank_max'] = rank_intervals(other[score_column])
            stability.append(dict(**identity, variant=Path(comparison).name, **compare_rankings(sub, other, top_k, key)))
    if not ranked:
        raise ValueError('empty score table')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(ranked, ignore_index=True).to_csv(output / 'rankings.tsv', sep='\t', index=False)
    if recalls:
        pd.DataFrame(recalls).to_csv(output / 'positive_recall.tsv', sep='\t', index=False)
    if stability:
        pd.DataFrame(stability).to_csv(output / 'robustness.tsv', sep='\t', index=False)
    return pd.DataFrame(recalls), pd.DataFrame(stability)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--positives')
    p.add_argument('--budgets', type=int, nargs='+')
    p.add_argument('--group-by', nargs='*')
    p.add_argument('--score-column', default='score')
    p.add_argument('--key', default='peptide')
    p.add_argument('--positive-column', default='label_screen')
    p.add_argument('--variant-columns', nargs='*', default=[])
    p.add_argument('--comparison')
    p.add_argument('--top-k', type=int, default=10)
    a = p.parse_args()
    analyze(a.scores, a.output, a.positives, a.budgets, a.group_by, a.score_column, a.key, a.positive_column, a.variant_columns, a.comparison, a.top_k)


if __name__ == '__main__':
    main()
