# Model architecture

CROSSTCR-SRC combines Shared, Ranking and Construct experts with a learned fusion gate.

| Component | Implementation |
| --- | --- |
| Pair encoder | [pair_model.py](../code/pair_model.py) and [expert_model.py](../code/expert_model.py) |
| Shared and Construct experts | [train_experts.py](../code/train_experts.py) |
| Ranking head and fusion gate | [fusion_core.py](../code/fusion_core.py) |
| Pair-model training | [train_pair.py](../code/train_pair.py) |
| Fusion-gate training | [train_fusion.py](../code/train_fusion.py) |

[config.json](config.json) records the network dimensions, training defaults, feature definitions and ensemble conventions. Pair-model pretraining uses five seeds: 42, 2026, 20260713, 31415 and 27182. SRC expert and gate training uses seeds 17, 29, 43, 71 and 101.

Shared learns recognition patterns across pairs. Construct adds a residual head for each registered TCR construct. Ranking combines a support–candidate similarity prior with a learned residual. The fusion gate combines normalized expert ranks and context features.

See [methodology](../docs/methodology.md) for the modeling framework and [downloads](../README.md#downloads) for the code archive.
