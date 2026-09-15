from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "cyp-challenge-train-test"
ANALYSIS_TABLES = PROJECT_ROOT / "analysis" / "cyp_pre_model" / "tables"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_CODE_VERSION = "openadmet-dmpnn-v1"
FEATURE_SCHEMA_VERSION = "openadmet-rdkit-2d-v1"

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
DIRECT_TARGETS = tuple(f"{cyp}_pIC50_direct_inhibition" for cyp in CYPS)
DIRECT_LOWER = tuple(f"{target}_conf_low" for target in DIRECT_TARGETS)
DIRECT_UPPER = tuple(f"{target}_conf_high" for target in DIRECT_TARGETS)
DIRECT_STD = tuple(f"{target}_std" for target in DIRECT_TARGETS)

FOLD_COLUMNS = {
    "ecfp_cluster": "ecfp_component_0.6",
    "ecfp_component_0.6": "ecfp_component_0.6",
    "ecfp_component_0.5": "ecfp_component_0.5",
    "random": "random",
    "scaffold": "scaffold_group",
    "scaffold_group": "scaffold_group",
}
