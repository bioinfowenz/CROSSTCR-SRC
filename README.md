# CROSSTCR-SRC

**Antigen-conditioned learning for T cell receptor cross-reactivity prediction**

CROSSTCR-SRC uses known antigen recognition to guide the prediction of T cell receptor (TCR) cross-reactivity. Given a TCR, a known recognized antigen and a set of candidate antigens, it combines shared recognition patterns, antigen relationships and construct-specific information to prioritize candidates for experimental testing.

## Model architecture

| Expert | Role |
| --- | --- |
| **Shared** | Learns recognition patterns shared across TCR–antigen pairs. |
| **Ranking** | Uses relationships between known and candidate antigens to guide candidate ordering. |
| **Construct** | Captures recognition patterns associated with individual TCR constructs. |

A learned fusion gate integrates the three experts. The primary modeling scope is human paired-chain TCRs and HLA class I–restricted antigen recognition.

## Repository structure

```text
CROSSTCR-SRC/
├── README.md
├── models/
│   ├── README.md
│   └── config.json
├── data/
│   ├── README.md
│   └── SOURCES.tsv
├── docs/
│   └── methodology.md
└── checksums/
    └── SHA256SUMS.txt
```

- [Models](models/README.md): model organization, configuration and download.
- [Data](data/README.md): source datasets, versions and download.
- [Methodology](docs/methodology.md): inputs, three experts and fusion.
- [Checksums](checksums/SHA256SUMS.txt): archive integrity verification.

## Downloads

| Resource | Download |
| --- | --- |
| Pretrained models | [CROSSTCR-SRC_weights_v1.0.zip](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_weights_v1.0.zip) |
| CROSSTCR and BATCAVE source datasets | [CROSSTCR-SRC_source_data_v1.0.zip](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_source_data_v1.0.zip) |

See the [v1.0 release](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/tag/v1.0) for the complete download list.

## Data sources

- [CROSSTCR](https://i.uestc.edu.cn/CROSSTCR/): curated TCR cross-reactivity records. [Resource publication](https://doi.org/10.1038/s41698-026-01644-2).
- [BATCAVE](https://github.com/meyer-lab-cshl/BATMAN-paper/tree/main/results_batman/tcr_epitope_datasets/mutational_scan_datasets/database): TCR–pMHC mutational-scan measurements.

Please cite the original resources when using their data. Source information and accompanying notices are included in the data archive.

## Questions and feedback

Please use the [Issues page](https://github.com/bioinfowenz/CROSSTCR-SRC/issues) for questions and feedback.
