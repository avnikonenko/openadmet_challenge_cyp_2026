from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
PHYS_CHEM_COLUMNS = (
    "MW", "cLogP", "TPSA", "Fsp3", "HBD", "HBA", "RotatableBonds", "AromaticRings"
)


def lightgbm_features(frame, skip_indices: set[int] | None = None) -> tuple[np.ndarray, list[str]]:
    skipped = skip_indices or set()
    values = np.zeros((len(frame), 2048), dtype=np.float32)
    skipped_positions = []
    for row_position, (row_index, smiles) in enumerate(frame["canonical_smiles"].items()):
        if row_index in skipped:
            skipped_positions.append(row_position)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {smiles}")
        DataStructs.ConvertToNumpyArray(MORGAN_GENERATOR.GetFingerprint(mol), values[row_position])
    descriptors = frame.loc[:, list(PHYS_CHEM_COLUMNS)].to_numpy(dtype=np.float32)
    if skipped_positions:
        descriptors[skipped_positions] = 0
    names = [f"ECFP6_{i}" for i in range(2048)] + list(PHYS_CHEM_COLUMNS)
    return np.concatenate([values, descriptors], axis=1), names


def _one_hot(value, choices) -> list[float]:
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


ATOM_NUMS = tuple(range(1, 101))
DEGREES = tuple(range(6))
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
CHIRAL_TAGS = tuple(range(4))
TOTAL_HS = tuple(range(5))
HYBRIDIZATIONS = (
    Chem.HybridizationType.S, Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2,
)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
BOND_STEREO = tuple(range(6))


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_NUMS)
        + _one_hot(atom.GetTotalDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(int(atom.GetChiralTag()), CHIRAL_TAGS)
        + _one_hot(atom.GetTotalNumHs(), TOTAL_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), atom.GetMass() * 0.01]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(int(bond.GetStereo()), BOND_STEREO)
    )


ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FDIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


@dataclass(frozen=True)
class MoleculeGraph:
    atom_features: np.ndarray
    bond_features: np.ndarray
    bond_sources: np.ndarray
    bond_destinations: np.ndarray
    reverse_bonds: np.ndarray


@lru_cache(maxsize=20_000)
def molecule_graph(smiles: str) -> MoleculeGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    atoms = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    features: list[list[float]] = []
    sources: list[int] = []
    destinations: list[int] = []
    reverses: list[int] = []
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feature = bond_features(bond)
        forward = len(features)
        reverse = forward + 1
        features.extend([bond_feature, bond_feature])
        sources.extend([left, right])
        destinations.extend([right, left])
        reverses.extend([reverse, forward])
    return MoleculeGraph(
        atom_features=atoms,
        bond_features=np.asarray(features, dtype=np.float32).reshape(-1, BOND_FDIM),
        bond_sources=np.asarray(sources, dtype=np.int64),
        bond_destinations=np.asarray(destinations, dtype=np.int64),
        reverse_bonds=np.asarray(reverses, dtype=np.int64),
    )


def calculated_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    return np.asarray(
        [
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.FractionCSP3(mol), Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
        ], dtype=np.float32,
    )
