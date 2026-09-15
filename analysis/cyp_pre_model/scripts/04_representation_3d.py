#!/usr/bin/env python3
"""Representation complementarity and multi-conformer sensitivity diagnostics."""

from __future__ import annotations

import gzip
import json
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pmapper.pharmacophore import Pharmacophore
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIGURES_DIR,
    MORGAN_GENERATOR,
    N_BITS,
    PMAPPER_CACHE_DIR,
    PMAPPER_TABLES_DIR,
    SEED,
    ensure_output_dirs,
    nearest_similarities,
    save_figure,
    stable_seed,
    valid_mol,
    write_csv,
)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


N_PAIR_SAMPLES = 20_000
N_SENSITIVITY_QUERIES_PER_BAND = 10
N_REFERENCE_RANDOM = 80
N_REFERENCE_MAX = 260
N_CONFORMERS = 5
PMAPPER_NCOMB = 4


def largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(fragments, key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms())), len(fragments)


def embed_one(name: str, smiles: str) -> dict[str, object]:
    result: dict[str, object] = {"Molecule_Name": name, "success": False, "error": "", "fragment_count_3d": 0, "optimization": "", "optimization_status": np.nan}
    try:
        mol, fragment_count = largest_fragment(valid_mol(smiles))
        result["fragment_count_3d"] = fragment_count
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = stable_seed(name)
        params.enforceChirality = True
        params.numThreads = 1
        params.timeout = 10
        status = AllChem.EmbedMolecule(mol_h, params)
        method = "ETKDGv3"
        if status != 0:
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol_h, params)
            method = "ETKDGv3_random_coordinates"
        result["embed_method"] = method
        if status != 0:
            raise ValueError(f"embedding failed with status {status}")
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            result["optimization"] = "MMFF94"
            result["optimization_status"] = int(AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500))
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            result["optimization"] = "UFF"
            result["optimization_status"] = int(AllChem.UFFOptimizeMolecule(mol_h, maxIters=500))
        else:
            result["optimization"] = "none"
        heavy = Chem.RemoveHs(mol_h)
        usr = list(rdMolDescriptors.GetUSR(heavy))
        usrcat = list(rdMolDescriptors.GetUSRCAT(heavy))
        result.update({f"USR_{i:02d}": value for i, value in enumerate(usr)})
        result.update({f"USRCAT_{i:02d}": value for i, value in enumerate(usrcat)})
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def generate_single_descriptors(train: pd.DataFrame, test: pd.DataFrame, workers: int) -> pd.DataFrame:
    cache = Path(__file__).resolve().parents[1] / "tables" / "single_conformer_3d_descriptors.csv"
    expected_names = list(train["Molecule_Name"].astype(str)) + list(test["Molecule_Name"].astype(str))
    if cache.exists():
        frame = pd.read_csv(cache)
        expected_seeds = [stable_seed(name) for name in expected_names]
        if (
            frame["Molecule_Name"].astype(str).tolist() == expected_names
            and len(frame) == len(expected_names)
            and "seed" in frame
            and frame["seed"].astype(int).tolist() == expected_seeds
        ):
            return frame
    items = [(str(row.Molecule_Name), str(row.canonical_smiles)) for row in pd.concat([train, test], ignore_index=True).itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda_pair_embed, items, chunksize=4))
    frame = pd.DataFrame(results)
    frame.insert(1, "split", ["train"] * len(train) + ["test"] * len(test))
    frame["seed"] = frame["Molecule_Name"].map(stable_seed)
    write_csv(frame, "single_conformer_3d_descriptors.csv")
    return frame


def lambda_pair_embed(item: tuple[str, str]) -> dict[str, object]:
    return embed_one(*item)


def descriptor_score(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Vectorized RDKit USR score, verified against GetUSRScore."""

    # RDKit divides the summed descriptor difference by 12 for both USR (12
    # values) and USRCAT (five 12-value feature blocks); it does not average over
    # all 60 USRCAT values.
    return 1.0 / (1.0 + np.sum(np.abs(references - query[None, :]), axis=1) / 12.0)


def load_pmapper_cache(split: str) -> tuple[list[str], dict[str, dict]]:
    path = PMAPPER_CACHE_DIR / f"pmapper_{split}_descriptor_cache.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = {str(row["Molecule_Name"]): row for row in payload["results"] if row["success"]}
    return [str(name) for name in payload["names"]], rows


def generalized_tanimoto_dict(left: dict[str, int], right: dict[str, int]) -> float:
    if len(left) > len(right):
        left, right = right, left
    dot = sum(float(value) * float(right.get(key, 0)) for key, value in left.items())
    left_norm = sum(float(value) ** 2 for value in left.values())
    right_norm = sum(float(value) ** 2 for value in right.values())
    denominator = left_norm + right_norm - dot
    return dot / denominator if denominator > 0 else 0.0


def bit_tanimoto_sets(left: set[int], right: set[int]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def nearest_3d(query: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    reference_values = reference[columns].to_numpy(float)
    scores = np.empty(len(query), dtype=np.float32)
    indices = np.empty(len(query), dtype=np.int32)
    for row_index, vector in enumerate(query[columns].to_numpy(float)):
        values = descriptor_score(vector, reference_values)
        best = int(np.argmax(values))
        scores[row_index] = values[best]
        indices[row_index] = best
    return scores, indices


def representation_complementarity(train: pd.DataFrame, test: pd.DataFrame, descriptors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    train_fps = [MORGAN_GENERATOR.GetFingerprint(valid_mol(value)) for value in train["canonical_smiles"]]
    test_fps = [MORGAN_GENERATOR.GetFingerprint(valid_mol(value)) for value in test["canonical_smiles"]]
    ecfp_scores, ecfp_indices = nearest_similarities(test_fps, train_fps)

    train_3d = descriptors.loc[(descriptors["split"] == "train") & descriptors["success"]].reset_index(drop=True)
    test_3d = descriptors.loc[(descriptors["split"] == "test") & descriptors["success"]].reset_index(drop=True)
    usr_columns = [column for column in descriptors if column.startswith("USR_")]
    usrcat_columns = [column for column in descriptors if column.startswith("USRCAT_")]
    usr_scores, usr_indices = nearest_3d(test_3d, train_3d, usr_columns)
    usrcat_scores, usrcat_indices = nearest_3d(test_3d, train_3d, usrcat_columns)

    pmapper = pd.read_csv(PMAPPER_TABLES_DIR / "pmapper_per_test_nearest_neighbors.csv")
    nearest = test[["Molecule_Name"]].copy()
    nearest["ECFP6_neighbor"] = train.iloc[ecfp_indices]["Molecule_Name"].to_numpy()
    nearest["ECFP6_score"] = ecfp_scores
    usr_map = pd.DataFrame({"Molecule_Name": test_3d["Molecule_Name"], "USR_neighbor": train_3d.iloc[usr_indices]["Molecule_Name"].to_numpy(), "USR_score": usr_scores, "USRCAT_neighbor": train_3d.iloc[usrcat_indices]["Molecule_Name"].to_numpy(), "USRCAT_score": usrcat_scores})
    nearest = nearest.merge(usr_map, on="Molecule_Name", how="left", validate="one_to_one")
    nearest = nearest.merge(pmapper[["Molecule_Name", "count_train_nn_name", "count_train_nn_similarity", "bit_train_nn_name", "bit_train_nn_similarity"]], on="Molecule_Name", how="left", validate="one_to_one")
    nearest = nearest.rename(columns={"count_train_nn_name": "pmapper_count_neighbor", "count_train_nn_similarity": "pmapper_count_score", "bit_train_nn_name": "pmapper_bit_neighbor", "bit_train_nn_similarity": "pmapper_bit_score"})
    write_csv(nearest, "representation_nearest_neighbors.csv", ["Molecule_Name"])

    descriptor_names = ("ECFP6", "USR", "USRCAT", "pmapper_count", "pmapper_bit")
    overlap_rows = []
    for left, right in combinations(descriptor_names, 2):
        valid = nearest[[f"{left}_neighbor", f"{right}_neighbor"]].dropna()
        score_valid = nearest[[f"{left}_score", f"{right}_score"]].dropna()
        rank = stats.spearmanr(score_valid.iloc[:, 0], score_valid.iloc[:, 1]) if len(score_valid) > 2 else None
        overlap_rows.append({"representation_1": left, "representation_2": right, "test_molecules_compared": len(valid), "same_nearest_neighbor_count": int((valid.iloc[:, 0] == valid.iloc[:, 1]).sum()), "same_nearest_neighbor_fraction": float((valid.iloc[:, 0] == valid.iloc[:, 1]).mean()), "nearest_score_spearman": float(rank.statistic) if rank else np.nan, "nearest_score_spearman_pvalue": float(rank.pvalue) if rank else np.nan})
    overlap_frame = pd.DataFrame(overlap_rows)
    write_csv(overlap_frame, "representation_nearest_neighbor_overlap.csv", ["representation_1", "representation_2"])

    _, train_pmapper = load_pmapper_cache("train")
    _, test_pmapper = load_pmapper_cache("test")
    train_3d_map = train_3d.set_index("Molecule_Name")
    test_3d_map = test_3d.set_index("Molecule_Name")
    common_train = sorted(set(train_3d_map.index) & set(train_pmapper))
    common_test = sorted(set(test_3d_map.index) & set(test_pmapper))
    train_index = {name: i for i, name in enumerate(train["Molecule_Name"].astype(str))}
    test_index = {name: i for i, name in enumerate(test["Molecule_Name"].astype(str))}
    rng = np.random.default_rng(SEED)
    sampled_train = rng.integers(0, len(common_train), size=N_PAIR_SAMPLES)
    sampled_test = rng.integers(0, len(common_test), size=N_PAIR_SAMPLES)
    pair_rows = []
    for pair_id, (test_choice, train_choice) in enumerate(zip(sampled_test, sampled_train)):
        test_name, train_name = common_test[test_choice], common_train[train_choice]
        test_pm, train_pm = test_pmapper[test_name], train_pmapper[train_name]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "test_name": test_name,
                "train_name": train_name,
                "ECFP6": DataStructs.TanimotoSimilarity(test_fps[test_index[test_name]], train_fps[train_index[train_name]]),
                "USR": descriptor_score(test_3d_map.loc[test_name, usr_columns].to_numpy(float), train_3d_map.loc[[train_name], usr_columns].to_numpy(float))[0],
                "USRCAT": descriptor_score(test_3d_map.loc[test_name, usrcat_columns].to_numpy(float), train_3d_map.loc[[train_name], usrcat_columns].to_numpy(float))[0],
                "pmapper_count": generalized_tanimoto_dict(test_pm["descriptors"], train_pm["descriptors"]),
                "pmapper_bit": bit_tanimoto_sets(set(test_pm["fingerprint"]), set(train_pm["fingerprint"])),
            }
        )
    sampled = pd.DataFrame(pair_rows)
    write_csv(sampled, "representation_sampled_pair_similarities.csv", ["pair_id"])
    correlation_rows = []
    for left, right in combinations(descriptor_names, 2):
        result = stats.spearmanr(sampled[left], sampled[right])
        correlation_rows.append({"representation_1": left, "representation_2": right, "sampled_pairs": len(sampled), "spearman": result.statistic, "pvalue": result.pvalue})
    correlation_frame = pd.DataFrame(correlation_rows)
    write_csv(correlation_frame, "representation_pairwise_rank_correlations.csv", ["representation_1", "representation_2"])

    overlap_matrix = pd.DataFrame(np.eye(len(descriptor_names)), index=descriptor_names, columns=descriptor_names)
    correlation_matrix = overlap_matrix.copy()
    for row in overlap_frame.itertuples(index=False):
        overlap_matrix.loc[row.representation_1, row.representation_2] = row.same_nearest_neighbor_fraction
        overlap_matrix.loc[row.representation_2, row.representation_1] = row.same_nearest_neighbor_fraction
    for row in correlation_frame.itertuples(index=False):
        correlation_matrix.loc[row.representation_1, row.representation_2] = row.spearman
        correlation_matrix.loc[row.representation_2, row.representation_1] = row.spearman
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.heatmap(overlap_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Same test-to-train nearest-neighbor fraction")
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=axes[1])
    axes[1].set_title(f"Pairwise similarity-rank correlation (N={len(sampled):,})")
    save_figure(fig, "representation_complementarity.png")

    summary_rows = []
    for descriptor_name in descriptor_names:
        values = nearest[f"{descriptor_name}_score"].dropna()
        summary_rows.append({"representation": descriptor_name, "test_molecules": len(values), "nearest_mean": values.mean(), "nearest_median": values.median(), "nearest_p05": values.quantile(0.05), "nearest_p95": values.quantile(0.95), "nearest_min": values.min(), "nearest_max": values.max()})
    write_csv(pd.DataFrame(summary_rows), "representation_nearest_score_summary.csv", ["representation"])

    metadata = pd.read_csv(PMAPPER_TABLES_DIR / "pmapper_descriptor_metadata.csv")
    train_metadata = metadata.loc[(metadata["split"] == "train") & metadata["success"]]
    selected_bits = nearest["pmapper_bit_neighbor"].map(train_metadata.set_index("Molecule_Name")["active_bits"])
    saturation = pd.DataFrame(
        [
            {"diagnostic": "train_fully_saturated_2048_bits", "count": int((train_metadata["active_bits"] >= N_BITS).sum()), "fraction": float((train_metadata["active_bits"] >= N_BITS).mean())},
            {"diagnostic": "train_at_least_90pct_saturated", "count": int((train_metadata["active_bits"] >= 0.9 * N_BITS).sum()), "fraction": float((train_metadata["active_bits"] >= 0.9 * N_BITS).mean())},
            {"diagnostic": "test_NNs_that_are_fully_saturated", "count": int((selected_bits >= N_BITS).sum()), "fraction": float((selected_bits >= N_BITS).mean())},
            {"diagnostic": "test_NNs_at_least_90pct_saturated", "count": int((selected_bits >= 0.9 * N_BITS).sum()), "fraction": float((selected_bits >= 0.9 * N_BITS).mean())},
        ]
    )
    write_csv(saturation, "pmapper_hashed_saturation_diagnostics.csv", ["diagnostic"])

    candidate_maps = {
        name: dict(zip(nearest["Molecule_Name"].astype(str), nearest[f"{name}_neighbor"].astype(str)))
        for name in descriptor_names
    }
    return nearest, candidate_maps


def embed_multiple(item: tuple[str, str, str]) -> dict[str, object]:
    split, name, smiles = item
    result: dict[str, object] = {"split": split, "Molecule_Name": name, "requested_conformers": N_CONFORMERS, "generated_conformers": 0, "success": False, "error": "", "conformers": []}
    try:
        mol, fragment_count = largest_fragment(valid_mol(smiles))
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = stable_seed(f"multi:{name}")
        params.enforceChirality = True
        params.numThreads = 1
        params.pruneRmsThresh = -1.0
        params.timeout = 10
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=N_CONFORMERS, params=params))
        if not conformer_ids:
            params.useRandomCoords = True
            conformer_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=N_CONFORMERS, params=params))
        result["generated_conformers"] = len(conformer_ids)
        result["fragment_count_3d"] = fragment_count
        if not conformer_ids:
            raise ValueError("no conformers embedded")
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            statuses = [int(value[0]) for value in AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500)]
            optimization = "MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            statuses = [int(value[0]) for value in AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=500)]
            optimization = "UFF"
        else:
            statuses = [-1] * len(conformer_ids)
            optimization = "none"
        heavy = Chem.RemoveHs(mol_h)
        conformer_rows = []
        for position, conformer_id in enumerate(conformer_ids):
            usr = list(rdMolDescriptors.GetUSR(heavy, confId=int(conformer_id)))
            usrcat = list(rdMolDescriptors.GetUSRCAT(heavy, confId=int(conformer_id)))
            one_conf = Chem.Mol(mol_h)
            source_conf = mol_h.GetConformer(int(conformer_id))
            one_conf.RemoveAllConformers()
            one_conf.AddConformer(source_conf, assignId=True)
            pharmacophore = Pharmacophore(bin_step=1, cached=True)
            pharmacophore.load_from_mol(one_conf)
            pmapper_counts = {str(key): int(value) for key, value in pharmacophore.get_descriptors(tol=0, ncomb=PMAPPER_NCOMB).items()}
            conformer_rows.append({"conformer_index": position, "optimization": optimization, "optimization_status": statuses[position], "USR": usr, "USRCAT": usrcat, "pmapper_count": pmapper_counts})
        result["conformers"] = conformer_rows
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def representation_similarity(left: object, right: object, representation: str) -> float:
    if representation == "pmapper_count":
        return generalized_tanimoto_dict(left, right)
    return float(1.0 / (1.0 + np.sum(np.abs(np.asarray(left) - np.asarray(right))) / 12.0))


def conformer_sensitivity(train: pd.DataFrame, test: pd.DataFrame, nearest: pd.DataFrame, candidate_maps: dict[str, dict[str, int]], workers: int) -> None:
    test_nn = nearest.set_index("Molecule_Name")["ECFP6_score"]
    bands = pd.cut(test_nn, bins=[-np.inf, 0.4, 0.5, 0.6, np.inf], labels=["low_<0.4", "moderate_0.4_0.5", "moderate_0.5_0.6", "high_ge_0.6"], right=False)
    rng = np.random.default_rng(SEED)
    selected_test = []
    for band in bands.cat.categories:
        names = sorted(bands.index[bands == band].astype(str))
        take = min(N_SENSITIVITY_QUERIES_PER_BAND, len(names))
        selected_test.extend(rng.choice(names, size=take, replace=False).tolist())
    selected_test = sorted(set(selected_test))

    reference_names: set[str] = set()
    for representation, mapping in candidate_maps.items():
        for name in selected_test:
            neighbor = mapping.get(name)
            if neighbor and neighbor != "nan":
                reference_names.add(str(neighbor))
    remaining = sorted(set(train["Molecule_Name"].astype(str)) - reference_names)
    extra = rng.choice(remaining, size=min(N_REFERENCE_RANDOM, len(remaining)), replace=False).tolist()
    reference_names.update(extra)
    if len(reference_names) > N_REFERENCE_MAX:
        mandatory = {candidate_maps[representation][name] for representation in candidate_maps for name in selected_test if name in candidate_maps[representation] and candidate_maps[representation][name] != "nan"}
        optional = sorted(reference_names - mandatory)
        keep_optional = max(0, N_REFERENCE_MAX - len(mandatory))
        reference_names = set(mandatory) | set(rng.choice(optional, size=min(keep_optional, len(optional)), replace=False).tolist())
    reference_names = sorted(reference_names)

    selection_rows = []
    for name in selected_test:
        selection_rows.append({"split": "test_query", "Molecule_Name": name, "selection_reason": str(bands.loc[name])})
    for name in reference_names:
        reasons = [representation for representation, mapping in candidate_maps.items() if name in {mapping.get(query) for query in selected_test}]
        selection_rows.append({"split": "train_reference", "Molecule_Name": name, "selection_reason": "NN:" + "|".join(reasons) if reasons else "fixed_seed_random"})
    write_csv(pd.DataFrame(selection_rows), "conformer_sensitivity_subset.csv", ["split", "Molecule_Name"])

    test_map = test.set_index("Molecule_Name")["canonical_smiles"]
    train_map = train.set_index("Molecule_Name")["canonical_smiles"]
    items = [("test", name, test_map.loc[name]) for name in selected_test] + [("train", name, train_map.loc[name]) for name in reference_names]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(embed_multiple, items, chunksize=1))
    result_map = {row["Molecule_Name"]: row for row in results if row["success"]}
    metadata_rows = []
    for row in results:
        statuses = [conf["optimization_status"] for conf in row["conformers"]]
        metadata_rows.append({key: row.get(key) for key in ("split", "Molecule_Name", "requested_conformers", "generated_conformers", "fragment_count_3d", "success", "error")} | {"optimization_not_converged": int(sum(value == 1 for value in statuses)), "seed": stable_seed(f"multi:{row['Molecule_Name']}")})
    write_csv(pd.DataFrame(metadata_rows), "conformer_sensitivity_embedding_metadata.csv", ["split", "Molecule_Name"])

    successful_queries = [name for name in selected_test if name in result_map]
    successful_references = [name for name in reference_names if name in result_map]
    detail_rows = []
    within_rows = []
    for name in successful_queries + successful_references:
        conformers = result_map[name]["conformers"]
        for representation in ("USR", "USRCAT", "pmapper_count"):
            values = [conf[representation] for conf in conformers]
            pair_scores = [representation_similarity(values[i], values[j], representation) for i, j in combinations(range(len(values)), 2)]
            within_rows.append({"split": result_map[name]["split"], "Molecule_Name": name, "representation": representation, "conformers": len(values), "conformer_pairs": len(pair_scores), "within_molecule_similarity_mean": np.mean(pair_scores), "within_molecule_similarity_SD": np.std(pair_scores, ddof=1) if len(pair_scores) > 1 else 0.0, "within_molecule_similarity_min": np.min(pair_scores), "within_molecule_similarity_max": np.max(pair_scores)})

    for query_name in successful_queries:
        query_conformers = result_map[query_name]["conformers"]
        for representation in ("USR", "USRCAT", "pmapper_count"):
            neighbors = []
            best_scores = []
            per_query_reference_scores: list[dict[str, float]] = []
            for query_conf in query_conformers:
                reference_scores = {}
                for reference_name in successful_references:
                    reference_scores[reference_name] = max(representation_similarity(query_conf[representation], reference_conf[representation], representation) for reference_conf in result_map[reference_name]["conformers"])
                best_name = max(reference_scores, key=lambda name: (reference_scores[name], name))
                neighbors.append(best_name)
                best_scores.append(reference_scores[best_name])
                per_query_reference_scores.append(reference_scores)
            modal_name, modal_count = Counter(neighbors).most_common(1)[0]
            modal_scores = [scores[modal_name] for scores in per_query_reference_scores]
            detail_rows.append({"Molecule_Name": query_name, "ECFP6_coverage_band": str(bands.loc[query_name]), "representation": representation, "query_conformers": len(query_conformers), "reference_molecules": len(successful_references), "unique_nearest_neighbors": len(set(neighbors)), "nearest_neighbor_changed": len(set(neighbors)) > 1, "modal_neighbor": modal_name, "modal_neighbor_fraction": modal_count / len(neighbors), "best_score_mean": np.mean(best_scores), "best_score_SD": np.std(best_scores, ddof=1) if len(best_scores) > 1 else 0.0, "best_score_range": np.ptp(best_scores), "modal_neighbor_score_SD": np.std(modal_scores, ddof=1) if len(modal_scores) > 1 else 0.0, "nearest_neighbors_by_conformer": "|".join(neighbors), "best_scores_by_conformer": "|".join(f"{value:.6f}" for value in best_scores)})

    details = pd.DataFrame(detail_rows)
    within = pd.DataFrame(within_rows)
    write_csv(details, "conformer_nearest_neighbor_sensitivity.csv", ["representation", "Molecule_Name"])
    write_csv(within, "within_molecule_conformer_similarity.csv", ["representation", "split", "Molecule_Name"])
    summary = details.groupby("representation").agg(queries=("Molecule_Name", "size"), NN_changed_count=("nearest_neighbor_changed", "sum"), NN_changed_fraction=("nearest_neighbor_changed", "mean"), median_unique_NNs=("unique_nearest_neighbors", "median"), median_modal_neighbor_fraction=("modal_neighbor_fraction", "median"), median_best_score_SD=("best_score_SD", "median"), p95_best_score_SD=("best_score_SD", lambda values: values.quantile(0.95)), median_best_score_range=("best_score_range", "median")).reset_index()
    write_csv(summary, "conformer_sensitivity_summary.csv", ["representation"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    sns.violinplot(data=details, x="representation", y="modal_neighbor_fraction", cut=0, inner="quartile", ax=axes[0])
    axes[0].set(title="Nearest-neighbor identity stability", xlabel="", ylabel="Modal NN fraction")
    sns.violinplot(data=details, x="representation", y="best_score_SD", cut=0, inner="quartile", ax=axes[1])
    axes[1].set(title="Nearest-score conformer sensitivity", xlabel="", ylabel="Score SD")
    sns.violinplot(data=within, x="representation", y="within_molecule_similarity_mean", cut=0, inner="quartile", ax=axes[2])
    axes[2].set(title="Within-molecule conformer similarity", xlabel="", ylabel="Mean pair similarity")
    save_figure(fig, "conformer_sensitivity.png")


def main() -> int:
    ensure_output_dirs()
    tables = Path(__file__).resolve().parents[1] / "tables"
    train = pd.read_csv(tables / "train_molecule_identity_and_properties.csv")
    test = pd.read_csv(tables / "test_molecule_identity_and_properties.csv")
    workers = min(8, max(1, (os_cpu_count() or 1)))
    print(f"Generating/reusing single-conformer descriptors with {workers} workers...", flush=True)
    descriptors = generate_single_descriptors(train, test, workers)
    # Validate the vectorized score formula against RDKit for both descriptor lengths.
    successful = descriptors.loc[descriptors["success"]]
    for prefix in ("USR_", "USRCAT_"):
        columns = [column for column in descriptors if column.startswith(prefix)]
        left, right = successful.iloc[0][columns].to_numpy(float), successful.iloc[1][columns].to_numpy(float)
        if not np.isclose(descriptor_score(left, right[None, :])[0], rdMolDescriptors.GetUSRScore(left.tolist(), right.tolist()), atol=1e-12):
            raise RuntimeError(f"Vectorized {prefix} score does not match RDKit")
    nearest, candidate_maps = representation_complementarity(train, test, descriptors)
    print("Running multi-conformer sensitivity subset...", flush=True)
    conformer_sensitivity(train, test, nearest, candidate_maps, workers)
    write_csv(pd.DataFrame([{"python": platform.python_version(), "single_conformer_rows": len(descriptors), "single_conformer_success": int(descriptors["success"].sum()), "pairwise_rank_sample": N_PAIR_SAMPLES, "sensitivity_query_target": 4 * N_SENSITIVITY_QUERIES_PER_BAND, "sensitivity_conformers": N_CONFORMERS}]), "representation_analysis_run_summary.csv")
    print("Representation and 3D sensitivity analysis complete", flush=True)
    return 0


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


if __name__ == "__main__":
    raise SystemExit(main())
