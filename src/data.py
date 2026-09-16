from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ANALYSIS_TABLES,
    CYPS,
    DATA_DIR,
    DIRECT_LOWER,
    DIRECT_STD,
    DIRECT_TARGETS,
    DIRECT_UPPER,
)
from .splits import auxiliary_inner_masks, split_masks


@dataclass
class DataBundle:
    frame: pd.DataFrame
    train_mask: pd.Series
    validation_mask: pd.Series
    task_names: tuple[str, ...]
    target_columns: tuple[str, ...]
    lower_columns: tuple[str, ...]
    upper_columns: tuple[str, ...]
    uncertainty_columns: tuple[str, ...]
    split_metadata: dict[str, object]
    scored_tasks: tuple[str, ...] | None = None
    task_labels: tuple[str, ...] | None = None

    @property
    def scored_task_names(self) -> tuple[str, ...]:
        """Tasks that are predicted, validated, and scored; defaults to every task.

        A joint multi-assay model trains on auxiliary assay columns as well, but is
        only ever scored on the direct-pIC50 tasks so its out-of-fold metrics stay
        comparable with the single-assay models.
        """
        return tuple(self.scored_tasks) if self.scored_tasks else tuple(self.task_names)

    @property
    def scored_indices(self) -> tuple[int, ...]:
        return tuple(self.task_names.index(task) for task in self.scored_task_names)

    @property
    def scored_labels(self) -> tuple[str, ...]:
        """CYP name reported for each scored task in predictions and metrics."""
        labels = tuple(self.task_labels) if self.task_labels else tuple(self.task_names)
        return tuple(labels[index] for index in self.scored_indices)


def _identities_and_folds() -> pd.DataFrame:
    identities = pd.read_csv(ANALYSIS_TABLES / "train_molecule_identity_and_properties.csv")
    folds = pd.read_csv(ANALYSIS_TABLES / "cv_fold_assignments.csv")
    if identities["Molecule_Name"].duplicated().any() or folds["Molecule_Name"].duplicated().any():
        raise ValueError("Canonical identity or fold assignment contains duplicate molecule IDs")
    identity_smiles = identities.set_index("Molecule_Name")["canonical_smiles"]
    fold_smiles = folds.set_index("Molecule_Name")["canonical_smiles"]
    if set(identity_smiles.index) != set(fold_smiles.index):
        raise ValueError("Canonical identity and fold tables contain different molecule IDs")
    common = identity_smiles.index.sort_values()
    if not identity_smiles.loc[common].equals(fold_smiles.loc[common]):
        raise ValueError("Canonical identities differ between identity and fold tables")
    joined = identities.merge(
        folds.drop(columns=["canonical_smiles", "scaffold_smiles"]),
        on="Molecule_Name", how="inner", validate="one_to_one",
    )
    if len(joined) != len(identities):
        raise ValueError("Existing folds do not cover every canonical training molecule")
    return joined


def _validate_source_smiles(
    identities: pd.DataFrame, source: pd.DataFrame, source_name: str,
    require_exact_ids: bool = False,
) -> None:
    """Ensure derived identities still correspond to the raw dataset being loaded."""
    required = {"Molecule_Name", "SMILES"}
    if not required.issubset(source):
        raise ValueError(f"{source_name} lacks columns: {sorted(required - set(source))}")
    raw = source.loc[:, ["Molecule_Name", "SMILES"]].drop_duplicates()
    if raw["Molecule_Name"].duplicated().any():
        bad = raw.loc[raw["Molecule_Name"].duplicated(False), "Molecule_Name"].unique()
        raise ValueError(
            f"{source_name} maps molecule IDs to multiple raw SMILES: {bad[:5].tolist()}"
        )
    identity_raw = identities.loc[:, ["Molecule_Name", "SMILES"]]
    if identity_raw["Molecule_Name"].duplicated().any():
        raise ValueError("Canonical identity table contains duplicate molecule IDs")
    compared = identity_raw.merge(
        raw, on="Molecule_Name", how="outer", suffixes=("_identity", "_source"),
        indicator=True, validate="one_to_one",
    )
    missing = compared["_merge"].eq("right_only")
    if require_exact_ids:
        missing |= compared["_merge"].eq("left_only")
    mismatched = compared["_merge"].eq("both") & compared["SMILES_identity"].ne(
        compared["SMILES_source"]
    )
    if missing.any() or mismatched.any():
        examples = compared.loc[
            missing | mismatched,
            ["Molecule_Name", "SMILES_identity", "SMILES_source", "_merge"],
        ].head(5).to_dict("records")
        raise ValueError(
            f"{source_name} no longer matches the derived molecule identity table; "
            f"unmatched={int(missing.sum())}, SMILES_mismatches={int(mismatched.sum())}, "
            f"examples={examples}"
        )


def load_direct_data(scheme: str, fold: int, cyps: list[str] | None = None) -> DataBundle:
    requested = tuple(cyps or CYPS)
    unknown = set(requested) - set(CYPS)
    if unknown:
        raise ValueError(f"Unknown CYP targets: {sorted(unknown)}")
    base = _identities_and_folds()
    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv")
    _validate_source_smiles(base, direct, "Direct-pIC50 dataset")
    frame = base.merge(direct.drop(columns="SMILES"), on="Molecule_Name", how="inner", validate="one_to_one")
    expected = set(direct["Molecule_Name"])
    if set(frame["Molecule_Name"]) != expected:
        raise ValueError("Direct-pIC50 rows failed canonical-identity/fold matching")
    indices = [CYPS.index(cyp) for cyp in requested]
    targets = tuple(DIRECT_TARGETS[i] for i in indices)
    lowers = tuple(DIRECT_LOWER[i] for i in indices)
    uppers = tuple(DIRECT_UPPER[i] for i in indices)
    uncertainties = tuple(DIRECT_STD[i] for i in indices)
    train_mask, validation_mask = split_masks(frame, scheme, fold)
    observed = frame.loc[:, list(targets)].notna().any(axis=1)
    train_mask &= observed
    validation_mask &= observed
    return DataBundle(
        frame=frame, train_mask=train_mask, validation_mask=validation_mask,
        task_names=requested, target_columns=targets, lower_columns=lowers,
        upper_columns=uppers, uncertainty_columns=uncertainties,
        split_metadata={"scheme": scheme, "outer_fold": fold, "purpose": "direct_pic50"},
    )


def _single_concentration_columns(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot the long single-concentration screen into one column per CYP."""
    single = pd.read_csv(DATA_DIR / "cyp-challenge-single-concentration-TRAIN.csv")
    _validate_source_smiles(base, single, "Single-concentration dataset")
    if single.duplicated(["Molecule_Name", "enzyme"]).any():
        raise ValueError("Single-concentration data contain duplicate molecule/CYP rows")
    values = single.pivot(index="Molecule_Name", columns="enzyme", values="log2fc_estimate")
    errors = single.pivot(index="Molecule_Name", columns="enzyme", values="log2fc_std_error")
    values = values.reindex(columns=CYPS).rename(columns={cyp: f"{cyp}_single_conc" for cyp in CYPS})
    errors = errors.reindex(columns=CYPS).rename(columns={cyp: f"{cyp}_single_conc_SE" for cyp in CYPS})
    return values, errors, single


def load_joint_multiassay_data(
    scheme: str, fold: int, cyps: list[str] | None = None
) -> DataBundle:
    """One molecule table carrying both assays as separately masked targets.

    Direct pIC50 and single-concentration log2 fold change become independent target
    columns of the same masked multitask vector, so a shared encoder learns from both
    at once instead of sequentially. Fold safety is identical to the direct task: every
    outer-validation molecule is withheld from training in *both* assays, and only the
    direct-pIC50 tasks are scored.
    """
    requested = tuple(cyps or CYPS)
    unknown = set(requested) - set(CYPS)
    if unknown:
        raise ValueError(f"Unknown CYP targets: {sorted(unknown)}")
    base = _identities_and_folds()
    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv")
    _validate_source_smiles(base, direct, "Direct-pIC50 dataset")
    values, errors, _single = _single_concentration_columns(base)
    frame = base.merge(
        direct.drop(columns="SMILES"), on="Molecule_Name", how="left", validate="one_to_one"
    )
    frame = frame.merge(values.reset_index(), on="Molecule_Name", how="left", validate="one_to_one")
    frame = frame.merge(errors.reset_index(), on="Molecule_Name", how="left", validate="one_to_one")
    if len(frame) != len(base):
        raise ValueError("Joint multi-assay assembly changed the molecule count")
    indices = [CYPS.index(cyp) for cyp in requested]
    direct_targets = tuple(DIRECT_TARGETS[index] for index in indices)
    single_targets = tuple(f"{cyp}_single_conc" for cyp in requested)
    task_names = tuple(f"{cyp}|direct_pic50" for cyp in requested) + tuple(
        f"{cyp}|single_concentration" for cyp in requested
    )
    target_columns = direct_targets + single_targets
    lower_columns = tuple(DIRECT_LOWER[index] for index in indices) + tuple("" for _ in requested)
    upper_columns = tuple(DIRECT_UPPER[index] for index in indices) + tuple("" for _ in requested)
    uncertainty_columns = tuple(DIRECT_STD[index] for index in indices) + tuple(
        f"{cyp}_single_conc_SE" for cyp in requested
    )
    direct_observed = frame.loc[:, list(direct_targets)].notna().any(axis=1)
    any_observed = frame.loc[:, list(target_columns)].notna().any(axis=1)
    train_mask, validation_mask = split_masks(frame, scheme, fold)
    train_mask &= any_observed
    validation_mask &= direct_observed
    return DataBundle(
        frame=frame, train_mask=train_mask, validation_mask=validation_mask,
        task_names=task_names, target_columns=target_columns,
        lower_columns=lower_columns, upper_columns=upper_columns,
        uncertainty_columns=uncertainty_columns,
        split_metadata={
            "scheme": scheme, "outer_fold": fold, "purpose": "joint_multiassay",
            "assays": ["direct_pic50", "single_concentration"],
        },
        scored_tasks=tuple(f"{cyp}|direct_pic50" for cyp in requested),
        task_labels=tuple(requested) * 2,
    )


def load_single_concentration_data(
    scheme: str, fold: int, inner_fold: int | None = None
) -> DataBundle:
    base = _identities_and_folds()
    values, errors, single = _single_concentration_columns(base)
    frame = base.merge(values.reset_index(), on="Molecule_Name", how="inner", validate="one_to_one")
    frame = frame.merge(errors.reset_index(), on="Molecule_Name", how="left", validate="one_to_one")
    expected = set(single["Molecule_Name"])
    if set(frame["Molecule_Name"]) != expected:
        raise ValueError("Auxiliary rows failed canonical-identity/fold matching")
    train_mask, validation_mask, selected_inner = auxiliary_inner_masks(
        frame, scheme, fold, inner_fold
    )
    targets = tuple(f"{cyp}_single_conc" for cyp in CYPS)
    uncertainty = tuple(f"{cyp}_single_conc_SE" for cyp in CYPS)
    empty_bounds = tuple("" for _ in CYPS)
    return DataBundle(
        frame=frame, train_mask=train_mask, validation_mask=validation_mask,
        task_names=CYPS, target_columns=targets, lower_columns=empty_bounds,
        upper_columns=empty_bounds, uncertainty_columns=uncertainty,
        split_metadata={
            "scheme": scheme, "outer_fold": fold, "inner_validation_fold": selected_inner,
            "purpose": "single_concentration_pretraining",
        },
    )


def load_test_frame() -> pd.DataFrame:
    frame = pd.read_csv(ANALYSIS_TABLES / "test_molecule_identity_and_properties.csv")
    blinded = pd.read_csv(DATA_DIR / "cyp-challenge-TEST-BLINDED.csv")
    _validate_source_smiles(frame, blinded, "Blinded test dataset", require_exact_ids=True)
    return frame


def matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(columns)].to_numpy(dtype=np.float32)


def save_split_ids(bundle: DataBundle, output_dir: Path) -> None:
    rows = []
    for role, mask in (("train", bundle.train_mask), ("validation", bundle.validation_mask)):
        selected = bundle.frame.loc[mask, ["Molecule_Name", "canonical_smiles"]].copy()
        selected.insert(0, "role", role)
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_dir / "split_molecules.csv", index=False)


def smoke_subset(bundle: DataBundle, train_size: int = 128, validation_size: int = 64) -> DataBundle:
    train_indices = bundle.frame.index[bundle.train_mask][:train_size]
    validation_indices = bundle.frame.index[bundle.validation_mask][:validation_size]
    bundle.train_mask = pd.Series(bundle.frame.index.isin(train_indices), index=bundle.frame.index)
    bundle.validation_mask = pd.Series(
        bundle.frame.index.isin(validation_indices), index=bundle.frame.index
    )
    bundle.split_metadata = {
        **bundle.split_metadata,
        "smoke_test": True,
        "smoke_train_size": int(bundle.train_mask.sum()),
        "smoke_validation_size": int(bundle.validation_mask.sum()),
    }
    return bundle
