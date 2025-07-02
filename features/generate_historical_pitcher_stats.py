# generate_historical_pitcher_stats.py

import os
import logging
from datetime import datetime, timedelta
from build_pitcher_stat_features import build_pitcher_stat_features

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MATCHUP_DIR = "C:/Users/roman/baseball_forecast_project/data/raw"
OUTPUT_DIR = "C:/Users/roman/baseball_forecast_project/data/processed"
DAYS_BACK = 30  # Set how far back you want to check

def get_target_dates():
    return [(datetime.today().date() - timedelta(days=i)) for i in range(1, DAYS_BACK + 1)]

def get_matchup_file(date_obj):
    return os.path.join(MATCHUP_DIR, f"mlb_probable_pitchers_{date_obj}.csv")

def get_output_file(date_obj):
    return os.path.join(OUTPUT_DIR, f"pitcher_stat_features_{date_obj}.csv")

def run_rolling_pitcher_stats():
    for date in get_target_dates():
        matchup_file = get_matchup_file(date)
        output_file = get_output_file(date)

        if not os.path.exists(matchup_file):
            logger.warning(f"Skipping {date} — matchup file not found.")
            continue

        if os.path.exists(output_file):
            logger.info(f"Skipping {date} — pitcher output already exists.")
            continue

        logger.info(f"Generating pitcher features for {date}")
        build_pitcher_stat_features(matchup_file)

# Standalone testable main block
if __name__ == "__main__":
    run_rolling_pitcher_stats()

    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_pitcher_stats.py --rolling 20