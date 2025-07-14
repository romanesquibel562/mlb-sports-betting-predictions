# features/generate_historical_pitcher_stats.py

import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to sys.path to allow root-level imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from features.build_pitcher_stat_features import build_pitcher_stat_features

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def get_target_dates(days_back):
    return [(datetime.today().date() - timedelta(days=i)) for i in range(1, days_back + 1)]

def get_matchup_file(date_obj):
    return RAW_DIR / f"mlb_probable_pitchers_{date_obj}.csv"

def get_output_file(date_obj):
    return PROCESSED_DIR / f"pitcher_stat_features_{date_obj}.csv"

def run_rolling_pitcher_stats(days_back: int = 30):
    for date in get_target_dates(days_back):
        matchup_file = get_matchup_file(date)
        output_file = get_output_file(date)

        if not matchup_file.exists():
            logger.warning(f"Skipping {date} — matchup file not found.")
            continue

        if output_file.exists():
            logger.info(f"Skipping {date} — pitcher output already exists.")
            continue

        logger.info(f"Generating pitcher features for {date}")
        build_pitcher_stat_features(matchup_file)

# Standalone CLI execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate historical pitcher stats over a rolling window.")
    parser.add_argument("--days_back", type=int, default=30, help="How many days back to generate pitcher stats for.")
    args = parser.parse_args()

    run_rolling_pitcher_stats(days_back=args.days_back)

    # Example run from terminal:
    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_pitcher_stats.py --rolling 20
