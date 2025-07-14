# features/generate_historical_batter_stats.py

import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
STATCAST_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOOKUP_PATH = PROJECT_ROOT / "utils" / "data" / "reference" / "batter_team_lookup.csv"

# Import feature builder
from features.build_batter_stat_features import build_batter_stat_features  # <--- LOCAL IMPORT

def get_existing_dates():
    return {
        f.stem.replace("batter_stat_features_", "")
        for f in OUTPUT_DIR.glob("batter_stat_features_*.csv")
    }

def run_rolling_batter_generator(n_days: int):
    existing = get_existing_dates()
    all_statcast_files = sorted(STATCAST_DIR.glob("statcast_*.csv"))
    processed = 0

    for fpath in all_statcast_files:
        date_str = fpath.stem.replace("statcast_", "")
        if date_str in existing:
            continue

        logger.info(f"Processing: {date_str}")
        try:
            statcast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            build_batter_stat_features(fpath, LOOKUP_PATH, statcast_date=statcast_date)
            processed += 1
        except Exception as e:
            logger.error(f"Failed to build batter features for {date_str}: {e}")

        if processed >= n_days:
            break

    logger.info(f"Done. Generated batter stats for {processed} new date(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=int, default=10, help="Number of new days to process.")
    args = parser.parse_args()

    run_rolling_batter_generator(args.rolling)

    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_batter_stats.py --rolling 20
