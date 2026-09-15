#!/usr/bin/env python3
"""Audit OpenADMET train/test similarity with DrrDom/pmapper descriptors.

The primary representation is pmapper's count-based four-feature descriptor
(`get_descriptors(ncomb=4)`) compared with generalized Tanimoto similarity.
A matched four-feature 2048-bit pmapper fingerprint is compared with binary
Tanimoto as a hashing sensitivity check.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import platform
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmapper
import scipy
from pmapper.pharmacophore import Pharmacophore
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy import sparse
from scipy.stats import spearmanr
from tqdm import tqdm


SEED = 20260824
N_BITS = 2048
N_COMB = 4
BIN_STEP = 1
TOLERANCE = 0
ACTIVATE_BITS = 1
EMBED_TIMEOUT_SECONDS = 10
OPTIMIZATION_MAX_ITERS = 500
TOP_N = 50

TRAIN_FILES = {
    "regression": "cyp-challenge-TRAIN_inhibition.csv",
    "tdi": "cyp-challenge-TRAIN_TDI.csv",
    "emax": "cyp-challenge-TRAIN_Emax.csv",
    "single": "cyp-challenge-single-concentration-TRAIN.csv",
}
TEST_FILE = "cyp-challenge-TEST-BLINDED.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def unique_molecules(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Molecule_Name", "SMILES"])
    conflicts = frame.groupby("Molecule_Name")["SMILES"].nunique()
    if (conflicts > 1).any():
        names = conflicts[conflicts > 1].index.tolist()
        raise ValueError(f"Conflicting SMILES for molecule names: {names[:5]}")
    return frame.drop_duplicates("Molecule_Name").reset_index(drop=True)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    sources = {name: unique_molecules(data_dir / fname) for name, fname in TRAIN_FILES.items()}
    tagged = []
    for source, frame in sources.items():
        tagged.append(frame.assign(first_source=source))
    train = pd.concat(tagged, ignore_index=True).drop_duplicates("Molecule_Name", keep="first")
    train = train.reset_index(drop=True)
    test = unique_molecules(data_dir / TEST_FILE)
    return train, test, sources


def largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    largest = max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))
    return largest, len(fragments)


def molecule_descriptor(item: tuple[str, str]) -> dict:
    name, smiles = item
    result = {
        "Molecule_Name": str(name),
        "success": False,
        "fragments": 0,
        "heavy_atoms": 0,
        "embed_method": "",
        "optimization": "",
        "optimization_status": None,
        "feature_count": 0,
        "feature_types": {},
        "descriptor_keys": 0,
        "descriptor_total": 0,
        "active_bits": 0,
        "error": "",
        "descriptors": {},
        "fingerprint": [],
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("SMILES parse failure")
        mol, n_fragments = largest_fragment(mol)
        result["fragments"] = n_fragments
        result["heavy_atoms"] = mol.GetNumHeavyAtoms()
        mol_h = Chem.AddHs(mol)

        seed = (zlib.crc32(str(name).encode("utf-8")) ^ SEED) & 0x7FFFFFFF
        seed = seed or 1
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.enforceChirality = True
        params.numThreads = 1
        params.timeout = EMBED_TIMEOUT_SECONDS
        status = AllChem.EmbedMolecule(mol_h, params)
        result["embed_method"] = "ETKDGv3"
        if status != 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.enforceChirality = True
            params.useRandomCoords = True
            params.numThreads = 1
            params.timeout = EMBED_TIMEOUT_SECONDS
            status = AllChem.EmbedMolecule(mol_h, params)
            result["embed_method"] = "ETKDGv3_random_coords"
        if status != 0:
            raise ValueError(f"embedding failed with status {status}")

        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            result["optimization"] = "MMFF94"
            result["optimization_status"] = int(
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=OPTIMIZATION_MAX_ITERS)
            )
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            result["optimization"] = "UFF"
            result["optimization_status"] = int(
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=OPTIMIZATION_MAX_ITERS)
            )
        else:
            result["optimization"] = "none"

        pharmacophore = Pharmacophore(bin_step=BIN_STEP, cached=True)
        pharmacophore.load_from_mol(mol_h)
        features = pharmacophore.get_features_count()
        descriptors = pharmacophore.get_descriptors(tol=TOLERANCE, ncomb=N_COMB)
        fingerprint = pharmacophore.get_fp(
            min_features=N_COMB,
            max_features=N_COMB,
            tol=TOLERANCE,
            nbits=N_BITS,
            activate_bits=ACTIVATE_BITS,
        )

        result["feature_types"] = {str(k): int(v) for k, v in features.items()}
        result["feature_count"] = int(sum(features.values()))
        result["descriptor_keys"] = len(descriptors)
        result["descriptor_total"] = int(sum(descriptors.values()))
        result["active_bits"] = len(fingerprint)
        result["descriptors"] = {str(k): int(v) for k, v in descriptors.items()}
        result["fingerprint"] = sorted(int(bit) for bit in fingerprint)
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def cache_payload_valid(payload: dict, names: list[str]) -> bool:
    settings = payload.get("settings", {})
    return payload.get("names") == names and settings == method_settings()


def method_settings() -> dict:
    return {
        "pmapper": getattr(pmapper, "__version__", "unknown"),
        "ncomb": N_COMB,
        "bin_step_angstrom": BIN_STEP,
        "tolerance": TOLERANCE,
        "nbits": N_BITS,
        "activate_bits": ACTIVATE_BITS,
        "seed": SEED,
        "embed_timeout_seconds_per_attempt": EMBED_TIMEOUT_SECONDS,
        "optimization_max_iters": OPTIMIZATION_MAX_ITERS,
    }


def generate_descriptors(
    frame: pd.DataFrame, cache_path: Path, split: str, workers: int
) -> list[dict]:
    names = frame["Molecule_Name"].astype(str).tolist()
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if cache_payload_valid(payload, names):
            return payload["results"]

    items = list(frame[["Molecule_Name", "SMILES"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(molecule_descriptor, items, chunksize=1),
                total=len(items),
                desc=f"pmapper descriptors ({split})",
            )
        )
    payload = {"settings": method_settings(), "names": names, "results": results}
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    return results


def successful_frame(frame: pd.DataFrame, results: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    ok = np.asarray([row["success"] for row in results], dtype=bool)
    return frame.loc[ok].reset_index(drop=True), [row for row in results if row["success"]]


def build_count_matrix(train_results: list[dict], test_results: list[dict]):
    vocabulary = sorted(
        {
            key
            for row in train_results + test_results
            for key in row["descriptors"].keys()
        }
    )
    index = {key: i for i, key in enumerate(vocabulary)}

    def build(rows: list[dict]) -> sparse.csr_matrix:
        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []
        for row_id, row in enumerate(rows):
            for key, value in row["descriptors"].items():
                row_ids.append(row_id)
                col_ids.append(index[key])
                values.append(float(value))
        return sparse.csr_matrix(
            (values, (row_ids, col_ids)),
            shape=(len(rows), len(vocabulary)),
            dtype=np.float32,
        )

    return build(train_results), build(test_results), vocabulary


def build_bit_vectors(results: list[dict]) -> tuple[list[ExplicitBitVect], np.ndarray]:
    vectors = []
    counts = np.empty(len(results), dtype=np.int32)
    for i, row in enumerate(results):
        vector = ExplicitBitVect(N_BITS)
        for bit in row["fingerprint"]:
            vector.SetBit(int(bit))
        vectors.append(vector)
        counts[i] = len(row["fingerprint"])
    return vectors, counts


def generalized_tanimoto_cross(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    norm_a = np.asarray(a.multiply(a).sum(axis=1)).ravel()
    norm_b = np.asarray(b.multiply(b).sum(axis=1)).ravel()
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    block = 32
    for start in tqdm(range(0, a.shape[0], block), desc="count test×train"):
        stop = min(start + block, a.shape[0])
        dot = (a[start:stop] @ b.T).toarray().astype(np.float32, copy=False)
        denominator = norm_a[start:stop, None] + norm_b[None, :] - dot
        np.divide(dot, denominator, out=out[start:stop], where=denominator > 0)
    return out


def generalized_tanimoto_within(matrix: sparse.csr_matrix):
    n = matrix.shape[0]
    norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    flat = np.empty(n * (n - 1) // 2, dtype=np.float32)
    nearest_score = np.full(n, -np.inf, dtype=np.float32)
    nearest_index = np.full(n, -1, dtype=np.int32)
    block = 32
    for start in tqdm(range(1, n, block), desc=f"count within n={n}"):
        stop = min(start + block, n)
        dot = (matrix[start:stop] @ matrix[:stop].T).toarray().astype(np.float32, copy=False)
        for local, i in enumerate(range(start, stop)):
            row_dot = dot[local, :i]
            denominator = norms[i] + norms[:i] - row_dot
            similarities = np.zeros(i, dtype=np.float32)
            np.divide(row_dot, denominator, out=similarities, where=denominator > 0)
            offset = i * (i - 1) // 2
            flat[offset : offset + i] = similarities
            best_j = int(np.argmax(similarities))
            nearest_score[i] = similarities[best_j]
            nearest_index[i] = best_j
            better = similarities > nearest_score[:i]
            better_index = np.flatnonzero(better)
            nearest_score[better_index] = similarities[better_index]
            nearest_index[better_index] = i
    return flat, nearest_score, nearest_index


def bit_tanimoto_cross(
    query: list[ExplicitBitVect], reference: list[ExplicitBitVect], query_counts: np.ndarray,
    reference_counts: np.ndarray,
) -> np.ndarray:
    out = np.empty((len(query), len(reference)), dtype=np.float32)
    for i, fingerprint in enumerate(tqdm(query, desc="fingerprint test×train")):
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, reference), dtype=np.float32)
        if query_counts[i] == 0:
            values[reference_counts == 0] = 0.0
        out[i] = values
    return out


def bit_tanimoto_within(vectors: list[ExplicitBitVect], counts: np.ndarray):
    n = len(vectors)
    flat = np.empty(n * (n - 1) // 2, dtype=np.float32)
    nearest_score = np.full(n, -np.inf, dtype=np.float32)
    nearest_index = np.full(n, -1, dtype=np.int32)
    for i in tqdm(range(1, n), desc=f"fingerprint within n={n}"):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(vectors[i], vectors[:i]), dtype=np.float32
        )
        if counts[i] == 0:
            similarities[counts[:i] == 0] = 0.0
        offset = i * (i - 1) // 2
        flat[offset : offset + i] = similarities
        best_j = int(np.argmax(similarities))
        nearest_score[i] = similarities[best_j]
        nearest_index[i] = best_j
        better = similarities > nearest_score[:i]
        better_index = np.flatnonzero(better)
        nearest_score[better_index] = similarities[better_index]
        nearest_index[better_index] = i
    return flat, nearest_score, nearest_index


def stat_dict(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def nearest_summary(values: np.ndarray) -> dict:
    summary = stat_dict(values)
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        summary[f"ge_{threshold:.1f}"] = float(np.mean(values >= threshold))
    return summary


def top_cross_pairs(
    matrix: np.ndarray, query: pd.DataFrame, reference: pd.DataFrame, metric: str
) -> pd.DataFrame:
    n = min(TOP_N, matrix.size)
    flat_index = np.argpartition(matrix.ravel(), -n)[-n:]
    flat_index = flat_index[np.argsort(matrix.ravel()[flat_index])[::-1]]
    query_index, reference_index = np.unravel_index(flat_index, matrix.shape)
    return pd.DataFrame(
        {
            "metric": metric,
            "query_name": query.iloc[query_index]["Molecule_Name"].to_numpy(),
            "query_smiles": query.iloc[query_index]["SMILES"].to_numpy(),
            "neighbor_name": reference.iloc[reference_index]["Molecule_Name"].to_numpy(),
            "neighbor_smiles": reference.iloc[reference_index]["SMILES"].to_numpy(),
            "similarity": matrix[query_index, reference_index],
        }
    )


def density_plot(
    ax, series: list[np.ndarray], labels: list[str], title: str, x_limit=(0.0, 1.0)
):
    bins = np.linspace(x_limit[0], x_limit[1], 81)
    for values, label in zip(series, labels):
        counts, edges = np.histogram(values, bins=bins)
        hist = counts / (len(values) * np.diff(edges))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, label=label, linewidth=1.6)
    ax.set(title=title, xlabel="Similarity", ylabel="Density", xlim=x_limit)
    ax.legend()
    ax.grid(alpha=0.25)


def save_plots(output_dir: Path, arrays: dict[str, np.ndarray], metadata: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    density_plot(
        axes[0, 0],
        [arrays["count_cross"], arrays["count_train"], arrays["count_test"]],
        ["test–train", "within train", "within test"],
        "pmapper count descriptors: all pairs (central range)",
        x_limit=(0, 0.08),
    )
    density_plot(
        axes[0, 1],
        [arrays["count_cross_nn"], arrays["count_train_nn"], arrays["count_test_nn"]],
        ["test→train NN", "train internal NN", "test internal NN"],
        "pmapper count descriptors: nearest neighbors",
    )
    density_plot(
        axes[1, 0],
        [arrays["bit_cross"], arrays["bit_train"], arrays["bit_test"]],
        ["test–train", "within train", "within test"],
        "pmapper 2048-bit fingerprint: all pairs",
    )
    density_plot(
        axes[1, 1],
        [arrays["bit_cross_nn"], arrays["bit_train_nn"], arrays["bit_test_nn"]],
        ["test→train NN", "train internal NN", "test internal NN"],
        "pmapper 2048-bit fingerprint: nearest neighbors",
    )
    fig.savefig(output_dir / "pmapper_similarity_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ["feature_count", "descriptor_keys", "active_bits"],
        ["Perceived pharmacophore features", "Unique count descriptors", "Active fingerprint bits"],
    ):
        for split, color in (("train", "tab:blue"), ("test", "tab:orange")):
            values = metadata.loc[metadata["split"] == split, column].to_numpy()
            ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.7, label=split, color=color)
        if column == "descriptor_keys":
            upper = float(np.quantile(metadata[column], 0.995))
            ax.set_xlim(0, upper)
            title += " (through 99.5th percentile)"
        ax.set(title=title, xlabel=column, ylabel="Density")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.savefig(output_dir / "pmapper_descriptor_coverage.png", dpi=180)
    plt.close(fig)


def save_agreement_plot(output_dir: Path, per_test: pd.DataFrame, train_bit_counts: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    saturation = per_test["bit_train_nn_active_bits"] / N_BITS
    points = axes[0].scatter(
        per_test["count_train_nn_similarity"],
        per_test["bit_train_nn_similarity"],
        c=saturation,
        cmap="viridis",
        s=22,
        alpha=0.75,
    )
    axes[0].set(
        title="Nearest-train similarity by representation",
        xlabel="Count-descriptor generalized Tanimoto",
        ylabel="2048-bit fingerprint Tanimoto",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.25)
    colorbar = fig.colorbar(points, ax=axes[0])
    colorbar.set_label("Active-bit fraction of fingerprint-selected neighbor")

    bins = np.linspace(0, N_BITS, 50)
    axes[1].hist(
        train_bit_counts,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label="all training molecules",
    )
    axes[1].hist(
        per_test["bit_train_nn_active_bits"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label="fingerprint-selected neighbors",
    )
    axes[1].set(
        title="Hashed-fingerprint saturation in selected neighbors",
        xlabel="Active bits (out of 2048)",
        ylabel="Density",
        xlim=(0, N_BITS),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "pmapper_representation_agreement.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    for directory in (cache_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    train, test, sources = load_data(args.data_dir)
    print(f"train={len(train)} test={len(test)} workers={args.workers}", flush=True)

    train_results_all = generate_descriptors(
        train, cache_dir / "pmapper_train_descriptor_cache.json.gz", "train", args.workers
    )
    test_results_all = generate_descriptors(
        test, cache_dir / "pmapper_test_descriptor_cache.json.gz", "test", args.workers
    )
    train_ok, train_results = successful_frame(train, train_results_all)
    test_ok, test_results = successful_frame(test, test_results_all)
    print(
        f"descriptor successes: train={len(train_ok)}/{len(train)} test={len(test_ok)}/{len(test)}",
        flush=True,
    )

    train_count, test_count, vocabulary = build_count_matrix(train_results, test_results)
    train_bits, train_bit_counts = build_bit_vectors(train_results)
    test_bits, test_bit_counts = build_bit_vectors(test_results)

    cross_count = generalized_tanimoto_cross(test_count, train_count)
    train_pairs_count, train_nn_count, train_nn_count_index = generalized_tanimoto_within(train_count)
    test_pairs_count, test_nn_count, test_nn_count_index = generalized_tanimoto_within(test_count)

    cross_bits = bit_tanimoto_cross(test_bits, train_bits, test_bit_counts, train_bit_counts)
    train_pairs_bits, train_nn_bits, train_nn_bits_index = bit_tanimoto_within(
        train_bits, train_bit_counts
    )
    test_pairs_bits, test_nn_bits, test_nn_bits_index = bit_tanimoto_within(test_bits, test_bit_counts)

    test_to_train_count_index = np.argmax(cross_count, axis=1)
    test_to_train_count = cross_count[np.arange(len(test_ok)), test_to_train_count_index]
    train_to_test_count = np.max(cross_count, axis=0)
    test_to_train_bit_index = np.argmax(cross_bits, axis=1)
    test_to_train_bits = cross_bits[np.arange(len(test_ok)), test_to_train_bit_index]
    train_to_test_bits = np.max(cross_bits, axis=0)

    per_test = test_ok[["Molecule_Name", "SMILES"]].copy()
    per_test["count_train_nn_name"] = train_ok.iloc[test_to_train_count_index][
        "Molecule_Name"
    ].to_numpy()
    per_test["count_train_nn_similarity"] = test_to_train_count
    per_test["count_test_nn_name"] = test_ok.iloc[test_nn_count_index]["Molecule_Name"].to_numpy()
    per_test["count_test_nn_similarity"] = test_nn_count
    per_test["bit_train_nn_name"] = train_ok.iloc[test_to_train_bit_index]["Molecule_Name"].to_numpy()
    per_test["bit_train_nn_similarity"] = test_to_train_bits
    per_test["bit_train_nn_active_bits"] = train_bit_counts[test_to_train_bit_index]
    per_test["count_similarity_of_bit_train_nn"] = cross_count[
        np.arange(len(test_ok)), test_to_train_bit_index
    ]
    per_test["bit_similarity_of_count_train_nn"] = cross_bits[
        np.arange(len(test_ok)), test_to_train_count_index
    ]
    per_test["bit_test_nn_name"] = test_ok.iloc[test_nn_bits_index]["Molecule_Name"].to_numpy()
    per_test["bit_test_nn_similarity"] = test_nn_bits
    per_test["same_count_bit_train_neighbor"] = (
        per_test["count_train_nn_name"] == per_test["bit_train_nn_name"]
    )
    per_test.to_csv(tables_dir / "pmapper_per_test_nearest_neighbors.csv", index=False)

    metadata_rows = []
    for split, results in (("train", train_results_all), ("test", test_results_all)):
        for row in results:
            metadata_rows.append(
                {
                    key: value
                    for key, value in {"split": split, **row}.items()
                    if key not in {"descriptors", "fingerprint", "feature_types"}
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(tables_dir / "pmapper_descriptor_metadata.csv", index=False)

    top = pd.concat(
        [
            top_cross_pairs(cross_count, test_ok, train_ok, "count_generalized_tanimoto"),
            top_cross_pairs(cross_bits, test_ok, train_ok, "bit_tanimoto"),
        ],
        ignore_index=True,
    )
    top.to_csv(tables_dir / "pmapper_top_test_train_pairs.csv", index=False)

    arrays = {
        "count_cross": cross_count.ravel(),
        "count_train": train_pairs_count,
        "count_test": test_pairs_count,
        "count_cross_nn": test_to_train_count,
        "count_train_nn": train_nn_count,
        "count_test_nn": test_nn_count,
        "bit_cross": cross_bits.ravel(),
        "bit_train": train_pairs_bits,
        "bit_test": test_pairs_bits,
        "bit_cross_nn": test_to_train_bits,
        "bit_train_nn": train_nn_bits,
        "bit_test_nn": test_nn_bits,
    }
    save_plots(figures_dir, arrays, metadata.loc[metadata["success"]].copy())
    save_agreement_plot(figures_dir, per_test, train_bit_counts)

    train_failures = [
        {key: row[key] for key in ("Molecule_Name", "embed_method", "error")}
        for row in train_results_all
        if not row["success"]
    ]
    test_failures = [
        {key: row[key] for key in ("Molecule_Name", "embed_method", "error")}
        for row in test_results_all
        if not row["success"]
    ]
    summary = {
        "method": {
            **method_settings(),
            "conformer": "largest fragment; explicit H; deterministic ETKDGv3; MMFF94, UFF fallback",
            "count_similarity": "generalized Tanimoto: dot(a,b)/(dot(a,a)+dot(b,b)-dot(a,b))",
            "bit_similarity": "binary Tanimoto on activated bit sets",
            "empty_empty_similarity": 0,
            "pair_comparisons": "exact for test-train, unique within-train, unique within-test, and nearest neighbors",
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "pmapper": getattr(pmapper, "__version__", "unknown"),
            "networkx": __import__("networkx").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "counts": {
            "train_union": len(train),
            "test": len(test),
            "train_sources": {key: len(value) for key, value in sources.items()},
            "train_success": len(train_ok),
            "test_success": len(test_ok),
            "train_failures": train_failures,
            "test_failures": test_failures,
            "descriptor_vocabulary": len(vocabulary),
            "train_zero_count_descriptors": int(np.sum(train_count.getnnz(axis=1) == 0)),
            "test_zero_count_descriptors": int(np.sum(test_count.getnnz(axis=1) == 0)),
            "train_zero_bit_fingerprints": int(np.sum(train_bit_counts == 0)),
            "test_zero_bit_fingerprints": int(np.sum(test_bit_counts == 0)),
            "train_optimization_not_converged": int(
                sum(row["optimization_status"] == 1 for row in train_results)
            ),
            "test_optimization_not_converged": int(
                sum(row["optimization_status"] == 1 for row in test_results)
            ),
        },
        "count_descriptors": {
            "test_train_all_pairs": stat_dict(cross_count.ravel()),
            "within_train_all_pairs": stat_dict(train_pairs_count),
            "within_test_all_pairs": stat_dict(test_pairs_count),
            "test_to_train_nearest": nearest_summary(test_to_train_count),
            "train_to_test_nearest": stat_dict(train_to_test_count),
            "within_train_nearest": stat_dict(train_nn_count),
            "within_test_nearest": stat_dict(test_nn_count),
            "test_internal_closer_than_train_fraction": float(
                np.mean(test_nn_count > test_to_train_count)
            ),
        },
        "bit_fingerprint": {
            "test_train_all_pairs": stat_dict(cross_bits.ravel()),
            "within_train_all_pairs": stat_dict(train_pairs_bits),
            "within_test_all_pairs": stat_dict(test_pairs_bits),
            "test_to_train_nearest": nearest_summary(test_to_train_bits),
            "train_to_test_nearest": stat_dict(train_to_test_bits),
            "within_train_nearest": stat_dict(train_nn_bits),
            "within_test_nearest": stat_dict(test_nn_bits),
            "test_internal_closer_than_train_fraction": float(
                np.mean(test_nn_bits > test_to_train_bits)
            ),
        },
        "representation_agreement": {
            "spearman_test_nearest_similarity": float(
                spearmanr(test_to_train_count, test_to_train_bits).statistic
            ),
            "same_nearest_train_fraction": float(per_test["same_count_bit_train_neighbor"].mean()),
        },
    }
    with (tables_dir / "pmapper_similarity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
