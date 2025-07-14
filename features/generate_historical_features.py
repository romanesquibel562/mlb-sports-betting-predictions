# features/generate_historical_features.py

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import modules (LOCAL imports — drop 'features.' prefix)
from scraping.scrape_historical_matchups import scrape_rolling_window
from features.generate_historical_team_form import run_team_form_rolling
from features.generate_historical_pitcher_stats import run_rolling_pitcher_stats
from features.generate_historical_batter_stats import run_rolling_batter_generator
from features.historical_main_features import build_historical_main_dataset

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_all_historical_features(days_back: int = 20):
    logger.info(f"Generating full historical feature set for the past {days_back} days...")

    # Step 1: Scrape matchups
    scrape_rolling_window(days_back)

    # Step 2: Generate team form
    run_team_form_rolling()

    # Step 3: Generate pitcher stats
    run_rolling_pitcher_stats()

    # Step 4: Generate batter stats
    run_rolling_batter_generator(days_back)

    # Step 5: Merge everything
    build_historical_main_dataset()

    logger.info("Finished generating full historical dataset.")

if __name__ == "__main__":
    generate_all_historical_features(days_back=20)

    # Run from terminal:
    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_features.py