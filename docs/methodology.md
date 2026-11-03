# Methodology

## Antigen-conditioned ranking

CROSSTCR-SRC uses a TCR's known antigen recognition to prioritize additional candidate antigens. Inputs comprise paired CDR3alpha and CDR3beta sequences, an HLA context, a known positive support peptide and candidate peptides. The released research variant uses one support peptide.

## Three experts

Shared learns recognition patterns across TCR-antigen pairs. Ranking uses support-candidate relationships. Construct captures recognition patterns associated with individual TCR constructs and uses its global head for an unregistered construct.

Frozen ESM-2 residue representations feed pair encoders. Ranking combines a frozen pair encoder, support-candidate descriptors and a BLOSUM62 similarity prior. The foundation model is facebook/esm2_t30_150M_UR50D; layer-30 residue representations exclude special tokens. [Configuration](../models/config.json) gives architecture and feature definitions.

## Fusion

Expert outputs are aggregated across five seeds and normalized to ranks within each candidate pool. Ranking orders by similarity prior and resolves prior ties using learned residuals. Five gates combine expert ranks, across-seed rank variation and context features; their scores are averaged.

Public expert order is Shared, Ranking, Construct; stored gate inputs use Construct, Shared, Ranking. Each gate uses its saved feature scaler. The research framework also applies a query-level novelty rule for dispatch to the Ranking branch. The released model component is the three-expert fusion ensemble.

The primary modeling scope is human paired-chain TCRs and HLA class I recognition. Candidate rankings support experimental prioritization.
