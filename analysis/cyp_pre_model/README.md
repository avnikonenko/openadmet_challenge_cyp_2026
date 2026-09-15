# CYP pre-model analysis

Current validation-oriented analysis before predictive-model development.

- `scripts/run_all.py`: deterministic entry point and output validator.
- `report.md`: concise findings and modelling implications.
- `tables/`: reproducible summary and molecule-level CSV outputs.
- `figures/`: generated PNG figures.
- `cache/`: matplotlib and reusable runtime cache files.

This product consumes the standalone pmapper artifacts in
`../descriptor_exploration/`; it does not train final prediction models.
