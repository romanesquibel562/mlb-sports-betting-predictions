# engineer_features.py

import pandas as pd
import os
import logging
import sys
from datetime import datetime

# Set up logging to print to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def engineer_features(statcast_path):
    logger.info(f"Starting feature engineering for: {statcast_path}")

    # Load Statcast data
    try:
        df = pd.read_csv(statcast_path)
        logger.info(f"Loaded {len(df)} rows from {statcast_path}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return None

    if df.empty:
        logger.warning("The dataset is empty. Skipping.")
        return None

    # Feature engineering
    df['spin_rate_drop'] = df['release_spin_rate'].pct_change(fill_method=None)
    df['velocity_drop'] = df['release_speed'].pct_change(fill_method=None)
    df.fillna(0, inplace=True)

    # Save to processed folder using date from filename
    output_dir = r"C:\Users\roman\baseball_forecast_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(statcast_path)  # e.g. statcast_2025-05-16.csv
    date_part = filename.replace("statcast_", "").replace(".csv", "")  # e.g. 2025-05-16
    output_path = os.path.join(output_dir, f"features_{date_part}.csv")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved engineered features to: {output_path}")

    return output_path

# Optional test block for direct runs
if __name__ == "__main__":
    test_path = r"C:\Users\roman\baseball_forecast_project\data\raw\statcast_2025-05-16.csv"
    engineer_features(test_path)

