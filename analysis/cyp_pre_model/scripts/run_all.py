#!/usr/bin/env python3
"""Run and validate the complete pre-model OpenADMET exploratory analysis."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
SCRIPTS = ANALYSIS / "scripts"
DESCRIPTOR_EXPLORATION = ROOT / "analysis" / "descriptor_exploration"
TABLES = ANALYSIS / "tables"
FIGURES = ANALYSIS / "figures"


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    print(f"\n>>> {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(ANALYSIS / "cache" / "mplconfig")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def validate_outputs() -> None:
    required_tables = {
        "analysis_summary.csv",
        "dataset_inventory.csv",
        "input_file_hashes.csv",
        "training_union_construction_counts.csv",
        "cross_source_name_identity_conflicts.csv",
        "cross_split_identity_overlap.csv",
        "target_distribution_summary.csv",
        "label_count_per_molecule.csv",
        "cross_cyp_correlations.csv",
        "uncertainty_summary.csv",
        "uncertainty_between_cyp_comparison.csv",
        "activity_cliff_threshold_statistics.csv",
        "scaffold_summary.csv",
        "per_cyp_chemical_space_coverage.csv",
        "auxiliary_assay_correlations.csv",
        "representation_pairwise_rank_correlations.csv",
        "conformer_sensitivity_summary.csv",
        "cv_fold_assignments.csv",
        "cv_split_summary.csv",
        "cv_official_mean_baseline.csv",
        "train_test_property_shift_statistics.csv",
    }
    missing = sorted(name for name in required_tables if not (TABLES / name).exists())
    if missing:
        raise FileNotFoundError(f"Missing required tables: {missing}")
    frames = {}
    for path in sorted(TABLES.glob("*.csv")):
        try:
            frames[path.name] = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV has no header/schema: {path}") from exc

    inventory = frames["dataset_inventory.csv"].set_index("dataset")
    if int(inventory.loc["train_union", "rows"]) != 6147 or int(inventory.loc["test", "rows"]) != 750:
        raise ValueError("Unexpected train/test molecule counts")
    if int(inventory[["invalid", "standardization_errors"]].to_numpy().sum()) != 0:
        raise ValueError("Molecule problems are present; inspect molecule_problems.csv")

    hashes = frames["input_file_hashes.csv"]
    for row in hashes.itertuples(index=False):
        path = ROOT / "cyp-challenge-train-test" / row.file
        if not path.exists() or path.stat().st_size != row.bytes:
            raise ValueError(f"Input provenance mismatch: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Input SHA-256 mismatch: {path}")

    cliff_counts = frames["activity_cliff_sample_counts.csv"]
    expected_pairs = cliff_counts["rows_after_valid_canonical_requirement"] * (
        cliff_counts["rows_after_valid_canonical_requirement"] - 1
    ) // 2
    if not expected_pairs.equals(cliff_counts["all_unique_pairs"]):
        raise ValueError("Activity-cliff pair counts are incomplete")

    assignments = frames["cv_fold_assignments.csv"]
    schemes = ["random", "scaffold_group", "ecfp_component_0.5", "ecfp_component_0.6"]
    if assignments["Molecule_Name"].nunique() != len(assignments):
        raise ValueError("CV assignments contain duplicate molecule names")
    for scheme in schemes:
        if set(assignments[scheme]) != set(range(5)) or assignments[scheme].isna().any():
            raise ValueError(f"Incomplete CV fold assignments for {scheme}")

    applicability = frames["cv_applicability_domain.csv"]
    if len(applicability) != len(assignments) * len(schemes):
        raise ValueError("CV applicability rows do not cover every molecule and scheme")
    if applicability.duplicated(["scheme", "Molecule_Name"]).any():
        raise ValueError("Duplicate CV applicability records")
    per_cyp_test = frames["test_per_cyp_applicability_domain.csv"]
    if len(per_cyp_test) != 4 * int(inventory.loc["test", "rows"]):
        raise ValueError("Per-CYP test applicability coverage is incomplete")

    nearest = frames["representation_nearest_neighbors.csv"]
    if len(nearest) != int(inventory.loc["test", "rows"]):
        raise ValueError("Representation nearest-neighbor coverage is incomplete")
    if len(frames["representation_sampled_pair_similarities.csv"]) != 20_000:
        raise ValueError("Representation rank-correlation sample is incomplete")
    conformer = frames["conformer_nearest_neighbor_sensitivity.csv"]
    if len(conformer) != 120 or conformer["query_conformers"].ne(5).any():
        raise ValueError("Conformer-sensitivity coverage is incomplete")

    summary = frames["analysis_summary.csv"]
    if summary.columns.tolist() != [
        "analysis",
        "main result",
        "implication for modelling",
        "priority",
    ]:
        raise ValueError("analysis_summary.csv does not have the requested schema")
    png_files = sorted(FIGURES.glob("*.png"))
    if not png_files:
        raise FileNotFoundError("No PNG figures were generated")
    for path in png_files:
        if path.stat().st_size < 1000:
            raise ValueError(f"Figure is unexpectedly small: {path}")
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Figure is not a valid PNG file: {path}")
    report = ANALYSIS / "report.md"
    if not report.exists() or report.stat().st_size < 5000:
        raise ValueError("analysis/report.md is missing or unexpectedly short")
    print(
        f"Validated {len(list(TABLES.glob('*.csv')))} CSV tables, "
        f"{len(png_files)} PNG figures, and {report}",
        flush=True,
    )


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pmapper_cache = DESCRIPTOR_EXPLORATION / "cache" / "pmapper_train_descriptor_cache.json.gz"
    if not pmapper_cache.exists():
        run(
            str(DESCRIPTOR_EXPLORATION / "scripts" / "pmapper_similarity_analysis.py"),
            str(ROOT / "cyp-challenge-train-test"),
            "--output-dir",
            str(ANALYSIS),
            "--workers",
            str(min(8, os.cpu_count() or 1)),
        )
    for script in (
        "01_core_eda.py",
        "02_activity_cliffs.py",
        "03_chemical_space_cv.py",
        "04_representation_3d.py",
        "05_make_report.py",
    ):
        run(script)
    validate_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
