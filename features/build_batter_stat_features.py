# build_batter_stat_features.py

import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path
import glob
from pybaseball import batting_stats
from unidecode import unidecode
import argparse

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Project Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOOKUP_PATH = BASE_DIR / "utils" / "data" / "reference" / "batter_team_lookup.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_file(directory: Path, prefix: str) -> Path:
    files = list(directory.glob(f"{prefix}_*.csv"))
    if not files:
        logger.error(f"No files found with prefix '{prefix}' in {directory}")
        return None
    return max(files, key=lambda x: x.stat().st_mtime)

#  Add statcast_date=None for compatibility
def build_batter_stat_features(statcast_path: Path, lookup_path: Path, statcast_date=None):
    try:
        logger.info(f"Loading Statcast data from: {statcast_path}")
        statcast_df = pd.read_csv(statcast_path)
        logger.info(f"Loaded Statcast: {len(statcast_df)} rows")

        logger.info(f"Loading batter lookup table from: {lookup_path}")
        lookup_df = pd.read_csv(lookup_path)
        logger.info(f"Loaded lookup table: {len(lookup_df)} players")

        lookup_df.columns = lookup_df.columns.str.strip()
        logger.info(f"Lookup columns after stripping: {list(lookup_df.columns)}")

        lookup_df = lookup_df.rename(columns={
            "batter": "mlbam_id",
            "team_name": "team"
        })
        lookup_df["lookup_player_name"] = lookup_df["name_first"].fillna('') + " " + lookup_df["name_last"].fillna('')

        merged = statcast_df.merge(lookup_df, left_on="batter", right_on="mlbam_id", how="inner")
        logger.info(f"Merged DataFrame: {len(merged)} rows")

        if merged.empty:
            logger.warning("No matching batter records found after merge.")
            return

        summary = merged.groupby(['mlbam_id', 'lookup_player_name', 'team']).agg(
            avg_launch_speed=('launch_speed', 'mean'),
            avg_bat_speed=('bat_speed', 'mean'),
            avg_swing_length=('swing_length', 'mean'),
            total_pitches=('pitch_number', 'count'),
            recent_home_runs=('events', lambda x: (x == 'home_run').sum()),
            recent_strikeouts=('events', lambda x: (x == 'strikeout').sum())
        ).reset_index()

        summary = summary.dropna(subset=[
            'avg_launch_speed', 'avg_bat_speed', 'avg_swing_length',
            'total_pitches', 'recent_home_runs', 'recent_strikeouts'
        ])
        logger.info(f"Remaining batters after dropping NaNs: {len(summary)}")

        year = datetime.today().year
        season_df = batting_stats(year)
        season_df = season_df[['Name', 'HR', 'SO', 'PA', 'AVG', 'SLG']]
        logger.info(f"Loaded full-season stats for {len(season_df)} players")

        summary['clean_name'] = summary['lookup_player_name'].apply(lambda x: unidecode(str(x)).strip().lower())
        season_df['clean_name'] = season_df['Name'].apply(lambda x: unidecode(str(x)).strip().lower())

        full = summary.merge(season_df[['clean_name', 'HR', 'SO', 'PA', 'AVG', 'SLG']], on='clean_name', how='left')
        full = full.dropna(subset=['HR', 'SO', 'PA', 'AVG', 'SLG'])

        full.drop(columns=['clean_name'], inplace=True)
        full = full.round(2)

        if statcast_date is None:
            statcast_date = datetime.today().date()

        output_path = PROCESSED_DIR / f"batter_stat_features_{statcast_date.strftime('%Y-%m-%d')}.csv"
        full.to_csv(output_path, index=False)
        logger.info(f"Saved batter stat features to: {output_path}")

    except Exception as e:
        logger.error(f"Error building batter stat features: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Statcast date to force output file format (YYYY-MM-DD)", type=str)
    args = parser.parse_args()

    statcast_path = find_latest_file(RAW_DIR, "statcast")

    if statcast_path and LOOKUP_PATH.exists():
        if args.date:
            statcast_date = pd.to_datetime(args.date).date()
        else:
            statcast_date = None
        build_batter_stat_features(statcast_path, LOOKUP_PATH, statcast_date=statcast_date)
    else:
        logger.error("Required input files not found.")
        
#     # cd C:\Users\roman\baseball_forecast_project\features
#     # python build_batter_stat_features.py
