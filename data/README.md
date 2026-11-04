# Source datasets

[Download the source-data archive](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_source_data_v1.0.zip).

| Resource | Files | Description |
| --- | --- | --- |
| CROSSTCR v1.4.0 | JSON and XLSX | Curated TCR cross-reactivity records and associated evidence |
| BATCAVE | MHC-I XLSX and MHC-II XLSX | TCR–pMHC mutational-scan measurements |

Both CROSSTCR formats are retained because their fields, missing values and numerical precision differ. They represent complementary views of the same resource. The archive preserves original source files and labels. The collections cover broader biological settings than the main human paired-chain HLA-I modeling scope; individual experimental analyses use selected records.

[SOURCES.tsv](SOURCES.tsv) lists source locations, local snapshot dates, sizes and file-level SHA-256 values. The archive includes accompanying notices and a checksum manifest. Cite the original resources when using these data:

- [CROSSTCR database](https://i.uestc.edu.cn/CROSSTCR/) and [publication](https://doi.org/10.1038/s41698-026-01644-2).
- [BATCAVE original database files](https://github.com/meyer-lab-cshl/BATMAN-paper/tree/main/results_batman/tcr_epitope_datasets/mutational_scan_datasets/database).

## AFP case-study data

[Download the AFP data archive](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_AFP_case_data_v1.0.zip).

The archive contains candidate and evidence tables, saved model scores, ranking summaries and robustness-analysis data for the AFP case study. Its file manifest identifies the corresponding analyses and source versions. [AFP analysis code](../code/afp_analysis.py) summarizes the published score tables.
