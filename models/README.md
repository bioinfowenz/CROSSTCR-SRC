# Pretrained models

CROSSTCR-SRC integrates Shared, Ranking and Construct experts through a learned fusion gate.

[Download the model package](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_weights_v1.0.zip).

The package contains shared/, ranking/, construct/ and gate/ directories, each with five seeds: 17, 29, 43, 71 and 101. The frozen Ranking pair encoder is stored in ranking/seed_17.pt and shared by all five Ranking heads.

[config.json](config.json) records the architecture, feature definitions, scaling, expert ordering and ensemble conventions.

## Version

The checkpoints correspond to research variant r03fold_18eb90111b0496f1f138 in CROSSTCR_SRC_v1_20261020, the original SRC variant underlying the AFP application. The archived parameters retain the original numerical values.

See [methodology](../docs/methodology.md) for inputs and fusion and [checksums](../checksums/SHA256SUMS.txt) for archive verification.
