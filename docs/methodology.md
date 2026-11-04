# Methodology

## Antigen-conditioned ranking

CROSSTCR-SRC uses a TCR's known antigen recognition to prioritize additional candidate antigens. Inputs comprise paired CDR3alpha and CDR3beta sequences, an HLA context, a known positive support peptide and candidate peptides.

## Three experts

Shared learns recognition patterns across TCR–antigen pairs. Ranking uses support–candidate relationships. Construct captures recognition patterns associated with individual TCR constructs and uses its global head for an unregistered construct.

Frozen ESM-2 residue representations feed pair encoders. Ranking combines pair representations, support–candidate descriptors and a BLOSUM62 similarity prior. The foundation model is `facebook/esm2_t30_150M_UR50D`; layer-30 residue representations exclude special tokens. [Configuration](../models/config.json) records architecture and feature definitions.

## Training and fusion

Pair-model training combines weighted binary cross-entropy and pairwise ranking. Shared and Construct training uses component-balanced panels and a binary cross-entropy objective with a pairwise ranking term. Ranking learns a support-conditioned residual from measured positive–negative pairs within prior-tie groups.

Fusion-gate training uses out-of-fold expert scores from the corresponding training membership. The same gate objective applies to familiar-receptor and cross-entity training settings. Training membership and support context are supplied for the selected setting.

Expert outputs are aggregated across five seeds and normalized to ranks within each candidate pool. Ranking orders by similarity prior and resolves prior ties using learned residuals. Five gates combine expert ranks, across-seed rank variation and six context features; their scores are averaged. The gate's expert order is retained in its saved training output.

## AFP analysis

[afp_analysis.py](../code/afp_analysis.py) reads existing score tables to summarize candidate ranks, known-positive retrieval and ranking robustness. Supplied rank intervals retain the source analysis's tie handling. Retrieval uses the expected fraction of a tied group selected at each integer budget. Robustness summaries include Spearman correlation and top-k Jaccard overlap, with all candidates tied at the top-k boundary retained.

The primary modeling scope is human paired-chain TCRs and HLA class I recognition. Candidate rankings support experimental prioritization.
