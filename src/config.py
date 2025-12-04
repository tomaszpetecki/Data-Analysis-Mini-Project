from pathlib import Path

#-------PATHS--------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_MIGRAINE_CSV = RAW_DATA_DIR / "migraine_dataset.csv"

#---COLUMNS---
USER_ID_COLUMN = "user_id"
DATE_COLUMN = "date"
TARGET_COLUMN = "migraine_occurrence"
VARS = ["sleep_hours","mood_level","stress_level","hydration_level","screen_time","migraine_occurrence","migraine_severity"]
TRUE_VARS = [i for i in VARS if i not in ["migraine_occurrence", "migraine_severity"]]

#-----MODELLING-----
RANDOM_SEED = 42
DEFAULT_SEQ_LENGTH = 7
ADDED_COLS = ["day_of_week","day_of_month","dow_sin","dow_cos","dom_sin","dom_cos"]

#-----APP-----
USER_ID = 3 #demo user id