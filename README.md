# CROSSTCR-SRC

CROSSTCR-SRC is an antigen-conditioned framework for predicting T cell receptor (TCR) cross-reactivity. It uses information about antigens already recognized by a TCR to rank additional candidate antigens and prioritize them for experimental testing.

## How it works

Given a TCR, a known positive antigen in its peptide–MHC context, and a set of candidate antigens, CROSSTCR-SRC combines three complementary experts:

- **Shared** learns recognition patterns shared across TCR–antigen pairs.
- **Ranking** uses the relationship between the known antigen and each candidate to guide candidate ordering.
- **Construct** captures recognition patterns associated with individual TCR constructs.

A learned fusion gate combines the expert scores into a candidate ranking. The main modeling scope is human paired-chain TCRs and HLA class I–restricted recognition. Predictions support experimental prioritization; ranking scores are not calibrated probabilities of recognition or toxicity.

## Downloads

Model weights and source-data files are distributed through [GitHub Releases](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/tag/v1.0).

| File | Contents |
| --- | --- |
| `CROSSTCR-SRC_weights_v1.0.zip` | Pretrained Shared, Ranking and Construct experts, fusion-gate weights, and the configuration needed to interpret the checkpoints. |
| `CROSSTCR-SRC_source_data_v1.0.zip` | Original CROSSTCR v1.4.0 JSON and XLSX files and BATCAVE MHC-I and MHC-II workbooks, with source information and checksums. |

The model package is a weights release rather than a standalone prediction application. Its README identifies the exact research version and checkpoint organization. The source-data archive preserves the original records and labels; it is not a processed training dataset or a release of experiment-specific data partitions. Inclusion in the archive does not mean that every source record was used or independently verified in this study. The source collections are broader than the model's primary scope.

## Data sources

- **CROSSTCR v1.4.0:** curated TCR cross-reactivity records. See the [CROSSTCR resource](https://i.uestc.edu.cn/CROSSTCR/) and its [publication](https://doi.org/10.1038/s41698-026-01644-2).
- **BATCAVE:** TCR–pMHC mutational-scan data. See the [original database files](https://github.com/meyer-lab-cshl/BATMAN-paper/tree/main/results_batman/tcr_epitope_datasets/mutational_scan_datasets/database).

Please cite the original resources when reusing their data. Source versions and accompanying notices are recorded inside the data archive.
