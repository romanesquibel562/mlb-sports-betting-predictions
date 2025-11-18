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

# Add statcast_date=None for compatibility
def build_batter_stat_features(statcast_path: Path, lookup_path: Path, statcast_date=None):
    try:
        logger.info(f"Loading Statcast data from: {statcast_path}")
        statcast_df = pd.read_csv(statcast_path)
        logger.info(f"Loaded Statcast: {len(statcast_df)} rows")

        # --- ensure expected columns exist / sane types ---
        # Some Statcast pulls may miss these engineered columns; create if missing
        for col in ['launch_speed', 'bat_speed', 'swing_length', 'pitch_number', 'events']:
            if col not in statcast_df.columns:
                statcast_df[col] = pd.NA
                logger.warning(f"Missing column '{col}' in Statcast; filling with NaN/empty.")
        # events should be string-like for startswith below
        statcast_df['events'] = statcast_df['events'].fillna('')

        logger.info(f"Loading batter lookup table from: {lookup_path}")
        lookup_df = pd.read_csv(lookup_path)
        logger.info(f"Loaded lookup table: {len(lookup_df)} players")

        # --- normalize lookup headers and keys ---
        lookup_df.columns = lookup_df.columns.str.strip()
        logger.info(f"Lookup columns after stripping: {list(lookup_df.columns)}")

        # Keep compatibility if older lookup still uses 'batter'
        if 'mlbam_id' not in lookup_df.columns and 'batter' in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={'batter': 'mlbam_id'})

        # Your lookup currently stores team abbreviations in 'team_name'
        if 'team' not in lookup_df.columns and 'team_name' in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={'team_name': 'team'})

        # Build a readable name for later merges
        if 'name_first' in lookup_df.columns and 'name_last' in lookup_df.columns:
            lookup_df["lookup_player_name"] = lookup_df["name_first"].fillna('') + " " + lookup_df["name_last"].fillna('')
        else:
            lookup_df["lookup_player_name"] = ""

        # --- align dtypes on merge keys ---
        # Statcast key is 'batter'; make numeric nullable int
        if 'batter' not in statcast_df.columns and 'mlbam_id' in statcast_df.columns:
            # tolerate alt key in Statcast if present
            statcast_df = statcast_df.rename(columns={'mlbam_id': 'batter'})
        if 'batter' not in statcast_df.columns:
            raise ValueError("Statcast data missing 'batter' (or 'mlbam_id') column for merge.")

        statcast_df['batter'] = pd.to_numeric(statcast_df['batter'], errors='coerce').astype('Int64')

        if 'mlbam_id' not in lookup_df.columns:
            raise ValueError("Lookup data missing 'mlbam_id' column after normalization.")
        lookup_df['mlbam_id'] = pd.to_numeric(lookup_df['mlbam_id'], errors='coerce').astype('Int64')
        lookup_df = lookup_df.dropna(subset=['mlbam_id']).drop_duplicates(subset=['mlbam_id'])

        # --- merge ---
        merged = statcast_df.merge(lookup_df, left_on="batter", right_on="mlbam_id", how="inner")
        logger.info(f"Merged DataFrame: {len(merged)} rows")

        if merged.empty:
            logger.warning("No matching batter records found after merge.")
            return

        # --- aggregate recent Statcast features per batter/team ---
        summary = merged.groupby(['mlbam_id', 'lookup_player_name', 'team']).agg(
            avg_launch_speed=('launch_speed', 'mean'),
            avg_bat_speed=('bat_speed', 'mean'),
            avg_swing_length=('swing_length', 'mean'),
            total_pitches=('pitch_number', 'count'),
            recent_home_runs=('events', lambda s: (s == 'home_run').sum()),
            # count all strikeout variants: strikeout, strikeout_double_play, etc.
            recent_strikeouts=('events', lambda s: s.str.startswith('strikeout', na=False).sum())
        ).reset_index()

        # drop rows where we couldn't compute the numeric summaries
        summary = summary.dropna(subset=[
            'avg_launch_speed', 'avg_bat_speed', 'avg_swing_length',
            'total_pitches', 'recent_home_runs', 'recent_strikeouts'
        ])
        logger.info(f"Remaining batters after dropping NaNs: {len(summary)}")

        # --- add season-long context ---
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

        # --- output ---
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
