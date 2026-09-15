#!/usr/bin/env python3
"""Exact ECFP6 activity-cliff analysis for each direct-inhibition endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CYPS,
    DATA_DIR,
    DIRECT_TARGETS,
    FIGURES_DIR,
    MORGAN_GENERATOR,
    ensure_output_dirs,
    save_figure,
    valid_mol,
    write_csv,
)

import matplotlib.pyplot as plt  # noqa: E402


SIMILARITY_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
BIN_EDGES = np.asarray([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001])


def all_pair_arrays(fingerprints: list, activities: np.ndarray, scaffolds: np.ndarray):
    n = len(fingerprints)
    pair_count = n * (n - 1) // 2
    similarities = np.empty(pair_count, dtype=np.float32)
    deltas = np.empty(pair_count, dtype=np.float32)
    same_scaffold = np.empty(pair_count, dtype=bool)
    left_indices = np.empty(pair_count, dtype=np.int32)
    right_indices = np.empty(pair_count, dtype=np.int32)
    offset = 0
    for i in range(1, n):
        width = i
        block = slice(offset, offset + width)
        similarities[block] = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        deltas[block] = np.abs(activities[i] - activities[:i])
        same_scaffold[block] = (scaffolds[i] == scaffolds[:i]) & (scaffolds[i] != "<ACYCLIC>")
        left_indices[block] = i
        right_indices[block] = np.arange(i, dtype=np.int32)
        offset += width
    return similarities, deltas, same_scaffold, left_indices, right_indices


def summary_row(cyp: str, scope: str, threshold: float, sims: np.ndarray, deltas: np.ndarray) -> dict[str, object]:
    selected = sims >= threshold
    values = deltas[selected]
    return {
        "CYP": cyp,
        "pair_scope": scope,
        "similarity_threshold": threshold,
        "pair_count": int(selected.sum()),
        "delta_mean": float(values.mean()) if len(values) else np.nan,
        "delta_median": float(np.median(values)) if len(values) else np.nan,
        "delta_Q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
        "delta_Q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
        "delta_IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if len(values) else np.nan,
        "cliff_delta_ge_1_count": int((values >= 1).sum()),
        "cliff_delta_ge_1_fraction": float((values >= 1).mean()) if len(values) else np.nan,
        "cliff_delta_ge_2_count": int((values >= 2).sum()),
        "cliff_delta_ge_2_fraction": float((values >= 2).mean()) if len(values) else np.nan,
    }


def binned_rows(cyp: str, scope: str, sims: np.ndarray, deltas: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        selected = (sims >= low) & (sims < high)
        values = deltas[selected]
        rows.append(
            {
                "CYP": cyp,
                "pair_scope": scope,
                "similarity_bin_low": low,
                "similarity_bin_high": min(high, 1.0),
                "similarity_bin_label": f"[{low:.1f}, {min(high, 1.0):.1f}{']' if high > 1 else ')'}",
                "pair_count": int(selected.sum()),
                "delta_mean": float(values.mean()) if len(values) else np.nan,
                "delta_median": float(np.median(values)) if len(values) else np.nan,
                "delta_Q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "delta_Q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                "delta_IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if len(values) else np.nan,
                "cliff_delta_ge_1_fraction": float((values >= 1).mean()) if len(values) else np.nan,
            }
        )
    return rows


def top_pair_rows(
    cyp: str,
    frame: pd.DataFrame,
    sims: np.ndarray,
    deltas: np.ndarray,
    same_scaffold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> pd.DataFrame:
    eligible = np.flatnonzero((sims >= 0.6) & (deltas >= 1.0))
    if not len(eligible):
        return pd.DataFrame()
    order = eligible[np.lexsort((-sims[eligible], -deltas[eligible]))][:100]
    rows = []
    for rank, pair_index in enumerate(order, start=1):
        i, j = int(left[pair_index]), int(right[pair_index])
        rows.append(
            {
                "CYP": cyp,
                "rank_by_delta_then_similarity": rank,
                "similarity": float(sims[pair_index]),
                "delta_pIC50": float(deltas[pair_index]),
                "same_nonacyclic_scaffold": bool(same_scaffold[pair_index]),
                "molecule_1": frame.iloc[i]["Molecule_Name"],
                "smiles_1": frame.iloc[i]["canonical_smiles"],
                "pIC50_1": frame.iloc[i]["pIC50"],
                "molecule_2": frame.iloc[j]["Molecule_Name"],
                "smiles_2": frame.iloc[j]["canonical_smiles"],
                "pIC50_2": frame.iloc[j]["pIC50"],
                "scaffold": frame.iloc[i]["scaffold"] if same_scaffold[pair_index] else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ensure_output_dirs()
    identity = pd.read_csv(Path(__file__).resolve().parents[1] / "tables" / "train_molecule_identity_and_properties.csv")
    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv").merge(
        identity[["Molecule_Name", "canonical_smiles", "scaffold"]],
        on="Molecule_Name",
        how="left",
        validate="one_to_one",
    )
    if direct["canonical_smiles"].isna().any():
        raise ValueError("Missing canonical identities; run 01_core_eda.py first")

    threshold_rows = []
    bin_rows = []
    top_frames = []
    count_rows = []
    plot_arrays = {}
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        frame = direct.loc[direct[target].notna(), ["Molecule_Name", "canonical_smiles", "scaffold", target]].copy().reset_index(drop=True)
        frame = frame.rename(columns={target: "pIC50"})
        fingerprints = [MORGAN_GENERATOR.GetFingerprint(valid_mol(smiles)) for smiles in frame["canonical_smiles"]]
        arrays = all_pair_arrays(fingerprints, frame["pIC50"].to_numpy(float), frame["scaffold"].to_numpy(str))
        sims, deltas, same_scaffold, left, right = arrays
        count_rows.append(
            {
                "CYP": cyp,
                "measured_rows_before_filtering": int(direct[target].notna().sum()),
                "rows_after_valid_canonical_requirement": len(frame),
                "unique_canonical_molecules": frame["canonical_smiles"].nunique(),
                "all_unique_pairs": len(sims),
                "same_nonacyclic_scaffold_pairs": int(same_scaffold.sum()),
                "different_or_acyclic_scaffold_pairs": int((~same_scaffold).sum()),
            }
        )
        for scope, mask in (
            ("all", np.ones(len(sims), dtype=bool)),
            ("same_nonacyclic_scaffold", same_scaffold),
            ("different_or_acyclic_scaffold", ~same_scaffold),
        ):
            for threshold in SIMILARITY_THRESHOLDS:
                threshold_rows.append(summary_row(cyp, scope, threshold, sims[mask], deltas[mask]))
            bin_rows.extend(binned_rows(cyp, scope, sims[mask], deltas[mask]))
        top = top_pair_rows(cyp, frame, sims, deltas, same_scaffold, left, right)
        if len(top):
            top_frames.append(top)
        plot_arrays[cyp] = (sims, deltas)
        print(f"{cyp}: molecules={len(frame)} pairs={len(sims):,}", flush=True)

    write_csv(pd.DataFrame(count_rows), "activity_cliff_sample_counts.csv", ["CYP"])
    write_csv(pd.DataFrame(threshold_rows), "activity_cliff_threshold_statistics.csv", ["CYP", "pair_scope", "similarity_threshold"])
    binned = pd.DataFrame(bin_rows)
    write_csv(binned, "activity_cliff_binned_statistics.csv", ["CYP", "pair_scope", "similarity_bin_low"])
    top_pairs = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    write_csv(top_pairs, "activity_cliff_top_pairs.csv", ["CYP", "rank_by_delta_then_similarity"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for ax, cyp in zip(axes.flat, CYPS):
        sims, deltas = plot_arrays[cyp]
        image = ax.hexbin(sims, deltas, gridsize=65, bins="log", mincnt=1, cmap="viridis")
        subset = binned.loc[(binned["CYP"] == cyp) & (binned["pair_scope"] == "all") & binned["pair_count"].gt(0)]
        x = (subset["similarity_bin_low"] + subset["similarity_bin_high"]) / 2
        ax.plot(x, subset["delta_median"], color="#f94144", marker="o", label="binned median")
        ax.fill_between(x, subset["delta_Q1"], subset["delta_Q3"], color="#f94144", alpha=0.2, label="binned IQR")
        ax.axhline(1, color="white", linestyle="--", alpha=0.8)
        ax.axvline(0.6, color="white", linestyle="--", alpha=0.8)
        ax.set(title=f"{cyp}: ECFP6 SAR smoothness", xlabel="Morgan radius-3 Tanimoto", ylabel="absolute delta pIC50", xlim=(0, 1))
        ax.legend(loc="upper left")
        fig.colorbar(image, ax=ax, label="log10 pair count")
    save_figure(fig, "activity_cliffs_ecfp6_hexbin.png")

    if len(top_pairs):
        selected = top_pairs.sort_values(["CYP", "same_nonacyclic_scaffold", "delta_pIC50", "similarity"], ascending=[True, False, False, False]).groupby("CYP", sort=False).head(1)
        mols = []
        legends = []
        for row in selected.itertuples(index=False):
            mols.extend([Chem.MolFromSmiles(row.smiles_1), Chem.MolFromSmiles(row.smiles_2)])
            legends.extend([f"{row.CYP} {row.molecule_1}\npIC50={row.pIC50_1:.2f}", f"sim={row.similarity:.2f}, delta={row.delta_pIC50:.2f}\n{row.molecule_2} pIC50={row.pIC50_2:.2f}"])
        image = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(420, 300), legends=legends, useSVG=False)
        image.save(FIGURES_DIR / "activity_cliff_examples.png")

    print("Activity-cliff analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
