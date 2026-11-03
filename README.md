# CROSSTCR-SRC

**Antigen-conditioned learning for T cell receptor cross-reactivity prediction**

CROSSTCR-SRC is a computational framework that uses known antigen recognition to guide the prediction of T cell receptor (TCR) cross-reactivity. Given a TCR, a known recognized antigen and a set of candidate antigens, CROSSTCR-SRC integrates complementary recognition signals to prioritize candidates for experimental testing.

## Overview

A known positive antigen provides a reference for exploring the recognition repertoire of a TCR. CROSSTCR-SRC incorporates this information into candidate ranking by combining shared TCR–antigen recognition patterns, relationships between known and candidate antigens, and construct-specific information.

The primary modeling scope is human paired-chain TCRs and HLA class I–restricted antigen recognition.

## Model architecture

CROSSTCR-SRC combines three complementary experts:

| Expert | Role |
| --- | --- |
| **Shared** | Learns recognition patterns shared across TCR–antigen pairs. |
| **Ranking** | Uses relationships between the known antigen and candidate antigens to guide candidate ordering. |
| **Construct** | Captures recognition patterns associated with individual TCR constructs. |

A learned fusion gate integrates the expert scores to produce a ranking within the candidate antigen set.

### Prediction setting

The framework considers:

- **TCR information:** paired CDR3α and CDR3β sequences.
- **Antigen context:** a known positive peptide and its MHC context.
- **Candidate antigens:** peptides to be evaluated for potential recognition.

The resulting candidate ranking supports the selection of antigens for follow-up functional experiments.

## Resources

Pretrained models and source datasets are available through the [v1.0 release](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/tag/v1.0).

| Resource | Description | Download |
| --- | --- | --- |
| **Pretrained models** | Shared, Ranking and Construct expert parameters, fusion-gate parameters, model configuration and version information. | [Model package](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_weights_v1.0.zip) |
| **Source datasets** | Original CROSSTCR v1.4.0 records and BATCAVE MHC-I and MHC-II mutational-scan workbooks, with source and version information. | [Data package](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/CROSSTCR-SRC_source_data_v1.0.zip) |
| **Checksums** | SHA-256 checksums for the model and data packages. | [Checksum file](https://github.com/bioinfowenz/CROSSTCR-SRC/releases/download/v1.0/SHA256SUMS.txt) |

## Model package

The model package contains five-seed ensembles of the three experts and their fusion gates, together with the frozen pair encoder used by the Ranking expert.

The accompanying configuration describes:

- Model architecture and parameter organization.
- Expert ordering and fusion inputs.
- Feature normalization and candidate-ranking conventions.
- Ensemble aggregation and external foundation-model information.

Refer to the README and configuration inside the archive for the released model version and checkpoint details.

## Source datasets

The data package preserves the original file formats and source labels.

### CROSSTCR

CROSSTCR organizes experimentally reported TCR cross-reactivity relationships and their associated evidence. The package includes both JSON and Excel representations of CROSSTCR v1.4.0, retaining the complementary information available in each format.

- [CROSSTCR database](https://i.uestc.edu.cn/CROSSTCR/)
- [CROSSTCR publication](https://doi.org/10.1038/s41698-026-01644-2)

### BATCAVE

BATCAVE compiles TCR–pMHC mutational-scan data describing responses to peptide variants. The package includes the original MHC-I and MHC-II workbooks.

- [BATCAVE source files](https://github.com/meyer-lab-cshl/BATMAN-paper/tree/main/results_batman/tcr_epitope_datasets/mutational_scan_datasets/database)
- [BATMAN project](https://github.com/meyer-lab-cshl/BATMAN)

Source versions, file-level checksums and accompanying notices are included in the data archive. Please cite the original resources when using their data.

## Intended use

CROSSTCR-SRC supports the exploration of TCR recognition repertoires and the prioritization of candidate antigens for experimental validation. Scores describe relative candidate ranking within the evaluated antigen set; functional recognition and off-target effects require experimental assessment.

## Questions and feedback

For questions about the released resources or to report an issue, please use the repository’s [Issues page](https://github.com/bioinfowenz/CROSSTCR-SRC/issues).
