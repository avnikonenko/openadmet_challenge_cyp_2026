# Analysis outputs

This directory is organized by analytical purpose.

- `cyp_pre_model/` contains the current, reproducible pre-model CYP challenge
  integrity, EDA, activity-cliff, chemical-space, CV, applicability-domain, and
  representation/3D analysis.
- `descriptor_exploration/` contains the earlier standalone pmapper descriptor and
  similarity audit, retained as an independent input to representation comparison.
- `ensemble_validation/` compares the selected three-family OOF ensemble on the
  frozen ECFP-cluster and scaffold splits, including paired uncertainty and
  applicability-domain diagnostics.

Within each product, `scripts/` contains executable analysis code, `tables/`
contains CSV/JSON outputs, `figures/` contains PNG figures, and `cache/` contains
reusable computational caches. Run the current complete analysis with:

```bash
conda run -n openadmet-cyp-sim python analysis/cyp_pre_model/scripts/run_all.py
```

The current interpretation is in `cyp_pre_model/report.md`.

Generated `tables/` and `figures/` are ignored by Git. Links to them in the detailed
report work after running the analysis locally, but not in the GitHub source view;
see the self-contained [public summary](../docs/data_analysis_summary.md) for key
numbers without generated files.
