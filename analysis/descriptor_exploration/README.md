# Descriptor exploration

Standalone pmapper count-descriptor and hashed-fingerprint similarity audit.

- `scripts/pmapper_similarity_analysis.py`: deterministic descriptor-generation and
  similarity-analysis script.
- `tables/`: descriptor metadata, nearest-neighbor tables, top pairs, and JSON
  summary.
- `figures/`: PNG similarity and descriptor diagnostics.
- `cache/`: gzip-compressed train/test descriptor caches reused by the current CYP
  pre-model representation analysis.
