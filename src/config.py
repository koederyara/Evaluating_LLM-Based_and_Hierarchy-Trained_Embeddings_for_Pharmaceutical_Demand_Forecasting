from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data paths ---
RAW_ATC = ROOT / "data" / "raw" / "ATC.csv"
DATA_ATC_PREPARED = ROOT / "data" / "prepared_data" / "data_atc.csv"
DATA_ATC_LINKS = ROOT / "data" / "prepared_data" / "data_atc_links.csv"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
EXPORTS_DIR = ROOT / "exports"

# --- Sales / forecasting data paths ---
# Pipeline stages: preparation (format only) -> exploration -> mapping -> modeling
# (value-level preprocessing). Blanks stay NaN until the modeling stage.
SALES_RAW_FORECAST_DIR = ROOT / "data" / "raw" / "forecasting"
SALES_RAW_EXPORT = SALES_RAW_FORECAST_DIR / "IQVIA_Finally we succeed.xlsx"
SALES_DIR = ROOT / "data" / "prepared_data" / "sales"
SALES_PREPARATION_DIR = SALES_DIR / "preparation"
SALES_EXPLORATION_DIR = SALES_DIR / "exploration"
SALES_EXPLORATION_DATASET_DIR = SALES_EXPLORATION_DIR / "dataset"  # explore_sales.py
SALES_EXPLORATION_SERIES_DIR = SALES_EXPLORATION_DIR / "series"    # investigate_series.py
SALES_EXPLORATION_SUBSETS_DIR = SALES_EXPLORATION_DIR / "subsets"  # analyze_subsets.py
SALES_MAPPING_DIR = SALES_DIR / "mapping"
SALES_MODELING_DIR = SALES_DIR / "modeling"
SALES_PREPARED_LONG = SALES_PREPARATION_DIR / "sales_prepared_long.csv"
SALES_PREPARED_WIDE = SALES_PREPARATION_DIR / "sales_prepared_wide.csv"

# --- Column names ---
COL_CLASS_ID = "Class ID"
COL_PREFERRED_LABEL = "Preferred Label"
COL_ATC_LEVEL = "ATC LEVEL"
COL_ONTOLOGY_TYPE = "ontology_type"

# --- Shared constants ---
RANDOM_SEED = 42

ATC_GROUPS: dict[str, str] = {
    "A": "Alimentary tract & metabolism",
    "B": "Blood & blood-forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system & sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Anti-infectives (systemic)",
    "L": "Antineoplastic & immunomodulating",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}

# --- Lorentz embedding hyperparameters ---
# Protocol follows the Nickel & Kiela (2018) Lorentz reference implementation
# (train-nouns.sh, facebookresearch/poincare-embeddings):
#   lr=0.5, burn_in=20, burn_in_multiplier=0.01, negs=50, epochs=1500
# Negatives are sampled uniformly (the paper's formal N(i,j) definition).
# batch_size=512 (not the reference's 50) is kept for throughput at ATC scale
# (~7K nodes); a paper-exact batch_size=50 run is far slower here and deferred.
LORENTZ_DIM = 10
LORENTZ_N_NEGATIVES = 50
LORENTZ_LR = 0.5
LORENTZ_EPOCHS = 1500
LORENTZ_BATCH_SIZE = 512
LORENTZ_SEED = 42
LORENTZ_BURN_IN_EPOCHS = 20    # warm-up phase, from train-nouns.sh
LORENTZ_BURN_IN_MULTIPLIER = 0.01  # lr during burn-in = lr * 0.01 = 0.005
