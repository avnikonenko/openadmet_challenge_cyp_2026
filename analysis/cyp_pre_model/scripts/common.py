"""Shared, deterministic utilities for the OpenADMET CYP exploratory analysis."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "cyp-challenge-train-test"
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ANALYSIS_DIR / "figures"
TABLES_DIR = ANALYSIS_DIR / "tables"
CACHE_DIR = ANALYSIS_DIR / "cache"
MPLCONFIG_DIR = CACHE_DIR / "mplconfig"
DESCRIPTOR_EXPLORATION_DIR = ROOT / "analysis" / "descriptor_exploration"
PMAPPER_CACHE_DIR = DESCRIPTOR_EXPLORATION_DIR / "cache"
PMAPPER_TABLES_DIR = DESCRIPTOR_EXPLORATION_DIR / "tables"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

SEED = 20260912
N_BITS = 2048
MORGAN_RADIUS = 3
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
DIRECT_TARGETS = tuple(f"{cyp}_pIC50_direct_inhibition" for cyp in CYPS)
TDI_TARGETS = ("CYP2D6_is_TDI", "CYP3A4_is_TDI")
TRAIN_FILES = {
    "direct": "cyp-challenge-TRAIN_inhibition.csv",
    "tdi": "cyp-challenge-TRAIN_TDI.csv",
    "emax": "cyp-challenge-TRAIN_Emax.csv",
    "single": "cyp-challenge-single-concentration-TRAIN.csv",
}
TEST_FILE = "cyp-challenge-TEST-BLINDED.csv"

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=N_BITS,
    includeChirality=False,
)


@dataclass(frozen=True)
class MoleculeRecord:
    """Canonical identity levels for one input SMILES."""

    valid: bool
    error: str
    standardization_error: str
    canonical_smiles: str | None
    nonisomeric_smiles: str | None
    fragment_parent_smiles: str | None
    charge_parent_smiles: str | None
    tautomer_parent_smiles: str | None
    fragment_count: int | None


def ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, TABLES_DIR, CACHE_DIR, MPLCONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(text: str, base: int = SEED) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little") & 0x7FFFFFFF
    return value or 1


@lru_cache(maxsize=None)
def _canonicalize_text(text: str) -> MoleculeRecord:
    """Canonicalize one nonempty SMILES; optional parent failures do not invalidate it."""

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return MoleculeRecord(
            False,
            "RDKit MolFromSmiles returned None",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
        )
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    fragments = len(Chem.GetMolFrags(mol))
    fragment_smiles = charge_smiles = tautomer_smiles = None
    standardization_errors: list[str] = []
    try:
        clean = rdMolStandardize.Cleanup(mol)
        fragment_parent = rdMolStandardize.FragmentParent(clean)
        fragment_smiles = Chem.MolToSmiles(
            fragment_parent, canonical=True, isomericSmiles=True
        )
    except Exception as exc:
        standardization_errors.append(f"fragment_parent={type(exc).__name__}: {exc}")
        fragment_parent = None
    if fragment_parent is not None:
        try:
            charge_parent = rdMolStandardize.ChargeParent(fragment_parent)
            charge_smiles = Chem.MolToSmiles(
                charge_parent, canonical=True, isomericSmiles=True
            )
        except Exception as exc:
            standardization_errors.append(f"charge_parent={type(exc).__name__}: {exc}")
            charge_parent = None
    else:
        charge_parent = None
    if charge_parent is not None:
        try:
            log_blocker = rdBase.BlockLogs()
            tautomer_parent = rdMolStandardize.TautomerParent(charge_parent)
            del log_blocker
            tautomer_smiles = Chem.MolToSmiles(
                tautomer_parent, canonical=True, isomericSmiles=True
            )
        except Exception as exc:
            if "log_blocker" in locals():
                del log_blocker
            standardization_errors.append(f"tautomer_parent={type(exc).__name__}: {exc}")
    return MoleculeRecord(
        True,
        "",
        "; ".join(standardization_errors),
        canonical,
        nonisomeric,
        fragment_smiles,
        charge_smiles,
        tautomer_smiles,
        fragments,
    )


def canonicalize_smiles(smiles: object) -> MoleculeRecord:
    if pd.isna(smiles):
        return MoleculeRecord(False, "missing SMILES", "", None, None, None, None, None, None)
    text = str(smiles).strip()
    if not text:
        return MoleculeRecord(False, "empty SMILES", "", None, None, None, None, None, None)
    try:
        return _canonicalize_text(text)
    except Exception as exc:  # preserve failures for reporting
        return MoleculeRecord(
            False,
            f"{type(exc).__name__}: {exc}",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
        )


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach identity columns without dropping or deduplicating rows."""

    cache: dict[str, MoleculeRecord] = {}
    records: list[MoleculeRecord] = []
    for value in frame["SMILES"]:
        key = "<MISSING>" if pd.isna(value) else str(value)
        if key not in cache:
            cache[key] = canonicalize_smiles(value)
        records.append(cache[key])
    result = frame.copy()
    fields = tuple(MoleculeRecord.__dataclass_fields__)
    for field in fields:
        result[field] = [getattr(record, field) for record in records]
    return result


def load_sources() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Return canonicalized sources, unique-name training union, and test data."""

    sources = {
        key: canonicalize_frame(pd.read_csv(DATA_DIR / filename))
        for key, filename in TRAIN_FILES.items()
    }
    tagged = []
    for source in ("direct", "tdi", "emax", "single"):
        unique_names = sources[source].drop_duplicates("Molecule_Name", keep="first")
        tagged.append(unique_names[["Molecule_Name", "SMILES"]].assign(first_source=source))
    train_union = canonicalize_frame(
        pd.concat(tagged, ignore_index=True).drop_duplicates("Molecule_Name", keep="first")
    )
    test = canonicalize_frame(pd.read_csv(DATA_DIR / TEST_FILE))
    return sources, train_union.reset_index(drop=True), test.reset_index(drop=True)


def valid_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Previously canonicalized SMILES failed to parse: {smiles}")
    return mol


def bemis_murcko_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "<ACYCLIC>"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def physicochemical_properties(mol: Chem.Mol) -> dict[str, float | str]:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "cLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "Fsp3": float(Descriptors.FractionCSP3(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": float(Lipinski.NumAromaticRings(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
        "scaffold": bemis_murcko_smiles(mol),
        "scaffold_atoms": float(
            MurckoScaffold.GetScaffoldForMol(mol).GetNumHeavyAtoms()
        ),
    }


def add_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add descriptors/fingerprints using the full canonical isomeric molecule."""

    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        if not row.valid:
            rows.append({})
            continue
        mol = valid_mol(row.canonical_smiles)
        rows.append(physicochemical_properties(mol))
    features = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame.copy(), features], axis=1)


def fingerprints(frame: pd.DataFrame) -> list:
    if not frame["valid"].all():
        bad = frame.loc[~frame["valid"], "Molecule_Name"].astype(str).tolist()
        raise ValueError(f"Cannot fingerprint invalid molecules: {bad[:10]}")
    return [MORGAN_GENERATOR.GetFingerprint(valid_mol(s)) for s in frame["canonical_smiles"]]


def nearest_similarities(query_fps: Sequence, reference_fps: Sequence) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_fps) == 0:
        raise ValueError("Reference fingerprint collection is empty")
    scores = np.empty(len(query_fps), dtype=np.float32)
    indices = np.empty(len(query_fps), dtype=np.int32)
    for index, fingerprint in enumerate(query_fps):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps),
            dtype=np.float32,
        )
        best = int(np.argmax(similarities))
        scores[index] = similarities[best]
        indices[index] = best
    return scores, indices


def within_nearest_similarities(fps: Sequence) -> tuple[np.ndarray, np.ndarray]:
    if len(fps) < 2:
        return np.full(len(fps), np.nan), np.full(len(fps), -1, dtype=np.int32)
    scores = np.full(len(fps), -np.inf, dtype=np.float32)
    indices = np.full(len(fps), -1, dtype=np.int32)
    for i in range(1, len(fps)):
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
        j = int(np.argmax(values))
        if values[j] > scores[i]:
            scores[i], indices[i] = values[j], j
        better = np.flatnonzero(values > scores[:i])
        scores[better] = values[better]
        indices[better] = i
    return scores, indices


def official_st_rae(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
) -> float:
    """Exact logic from official tutorial revision 858ae63c, without bootstrapping."""

    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    numerator = np.clip(pred - high, 0, None) + np.clip(low - pred, 0, None)
    baseline = float(np.mean(y))
    denominator = np.clip(baseline - high, 0, None) + np.clip(low - baseline, 0, None)
    total = float(np.sum(denominator))
    return float(np.sum(numerator) / total) if total > 0 else np.nan


def standardized_mean_difference(a: Iterable[float], b: Iterable[float]) -> float:
    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / max(len(left) + len(right) - 2, 1))
    return float((right.mean() - left.mean()) / pooled) if pooled > 0 else 0.0


def distribution_shift(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    return {
        "standardized_mean_difference_test_minus_train": standardized_mean_difference(left, right),
        "ks_statistic": float(stats.ks_2samp(left, right).statistic),
        "ks_pvalue": float(stats.ks_2samp(left, right).pvalue),
        "wasserstein_distance": float(stats.wasserstein_distance(left, right)),
    }


def write_csv(frame: pd.DataFrame, filename: str, sort_by: Sequence[str] | None = None) -> Path:
    ensure_output_dirs()
    output = frame.copy()
    if sort_by and len(output):
        missing_sort_columns = set(sort_by) - set(output.columns)
        if missing_sort_columns:
            raise ValueError(f"Cannot sort {filename}; missing columns: {sorted(missing_sort_columns)}")
        output = output.sort_values(list(sort_by), kind="stable")
    path = TABLES_DIR / filename
    output.to_csv(path, index=False)
    return path


def save_figure(fig: plt.Figure, filename: str, dpi: int = 180) -> Path:
    ensure_output_dirs()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def environment_table() -> pd.DataFrame:
    import matplotlib as mpl
    import scipy
    import sklearn

    return pd.DataFrame(
        [
            ("python", os.sys.version.split()[0]),
            ("rdkit", rdBase.rdkitVersion),
            ("numpy", np.__version__),
            ("pandas", pd.__version__),
            ("scipy", scipy.__version__),
            ("scikit-learn", sklearn.__version__),
            ("matplotlib", mpl.__version__),
        ],
        columns=["package", "version"],
    )
