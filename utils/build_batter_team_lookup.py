# build_batter_team_lookup.py

import pandas as pd
import os
from datetime import datetime
import logging
from pybaseball import playerid_reverse_lookup
from pathlib import Path

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Project Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = BASE_DIR / "utils" / "data" / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def build_batter_team_lookup(statcast_path: str) -> str:
    try:
        df = pd.read_csv(statcast_path)
        logger.info(f"Loaded Statcast data: {len(df)} rows")

        # Ensure required columns exist
        required_cols = ['batter', 'home_team', 'away_team', 'inning_topbot']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")

        # Subset and drop rows with missing batter or team info
        lookup = df[required_cols].dropna().copy()
        lookup['batter'] = lookup['batter'].astype(int)

        # Guess batter's team based on inning context
        lookup['team_name'] = lookup.apply(
            lambda row: row['home_team'] if row['inning_topbot'] == 'Bot' else row['away_team'],
            axis=1
        )

        # Drop duplicates
        batter_team_map = lookup[['batter', 'team_name']].drop_duplicates()

        # Add player names
        batter_ids = batter_team_map['batter'].tolist()
        player_info = playerid_reverse_lookup(batter_ids)
        player_info.rename(columns={'key_mlbam': 'batter'}, inplace=True)

        final_df = pd.merge(batter_team_map, player_info, on='batter', how='left')
        final_df = final_df[['batter', 'name_first', 'name_last', 'team_name']].dropna()

        # === Save both date-stamped and latest versions ===
        statcast_date = extract_date_from_filename(statcast_path)
        dated_output_path = REFERENCE_DIR / f"batter_team_lookup_{statcast_date}.csv"
        latest_output_path = REFERENCE_DIR / "batter_team_lookup.csv"

        final_df.to_csv(dated_output_path, index=False)
        final_df.to_csv(latest_output_path, index=False)

        logger.info(f"Saved dated lookup to: {dated_output_path}")
        logger.info(f"Overwrote latest lookup at: {latest_output_path}")
        return str(latest_output_path)

    except Exception as e:
        logger.error(f"Failed to build batter-team lookup: {e}")
        return None
    

def extract_date_from_filename(path: str) -> str:
    """Extracts date from a statcast_YYYY-MM-DD.csv filename."""
    try:
        filename = Path(path).name
        date_str = filename.replace("statcast_", "").replace(".csv", "")
        datetime.strptime(date_str, "%Y-%m-%d")  # validate format
        return date_str
    except Exception:
        return datetime.today().strftime("%Y-%m-%d")

# === Run standalone ===
if __name__ == "__main__":
    statcast_dir = BASE_DIR / "data" / "raw"
    statcast_files = sorted(statcast_dir.glob("statcast_*.csv"))
    
    if not statcast_files:
        logger.error("No statcast files found.")
    else:
        latest_statcast = statcast_files[-1]
        build_batter_team_lookup(latest_statcast)


# cd C:\Users\roman\baseball_forecast_project\utils
# python build_batter_team_lookup.py


