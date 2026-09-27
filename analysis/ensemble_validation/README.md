# Ensemble validation

Run from the repository root with saved cluster outputs copied to a readable path:

```bash
python analysis/ensemble_validation/scripts/validate_ensemble.py \
  --results-root /home/sany/Workspace/Projects/openadmet_models/openadmet/outputs
```

The script reads the three finished model families for each of the frozen ECFP-cluster
and scaffold CV schemes. It averages three seeds within each family, then equally
averages transfer D-MPNN, joint multi-assay D-MPNN, and LightGBM. It verifies OOF
keys, prediction scale, truth/bounds, and saved fold assignments. The project's
`src.metrics.regression_metrics` computes the challenge ST-RAE. ECFP6 similarity is
calculated against the saved direct-pIC50 training molecules for each fold using
Morgan radius 3, 2048 bits. The frozen assignment and scaffold identity come from
`analysis/cyp_pre_model/tables/cv_fold_assignments.csv`; the selected examples in
`activity_cliff_top_pairs.csv` are **not** an exhaustive cliff census.

CSV tables are written to `tables/`, PNG figures to `figures/`, and interpretation
to [report.md](report.md). Tables and figures are generated artifacts and are ignored
by Git. Use `--bootstrap` and `--seed` to change the paired molecule bootstrap.
No model training or blind-test scoring occurs.
