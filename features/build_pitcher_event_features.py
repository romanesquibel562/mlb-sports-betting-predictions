# build_pitcher_event_features.py

import pandas as pd
import os
import logging
import sys
from datetime import datetime
from pybaseball import playerid_reverse_lookup

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def build_pitcher_event_features(statcast_path):
    logger.info(f"Building pitcher-level features from: {statcast_path}")

    try:
        df = pd.read_csv(statcast_path)
        logger.info(f"Loaded {len(df)} rows from Statcast.")
    except Exception as e:
        logger.error(f"Failed to load Statcast file: {e}")
        return None

    if df.empty:
        logger.warning("Empty Statcast dataset, skipping pitcher feature generation.")
        return None

    # Filter rows where the player is a pitcher (they threw a pitch)
    df = df[df['player_type'] == 'pitcher'] if 'player_type' in df.columns else df[df['pitcher'].notna()]

    # Aggregated pitcher features
    summary = df.groupby('pitcher').agg({
        'release_speed': 'mean',
        'release_spin_rate': 'mean',
        'pitch_number': 'max',
        'description': lambda x: (x == 'strikeout').sum(),
        'events': lambda x: (x == 'walk').sum() if x.notna().any() else 0
    }).reset_index()

    summary.rename(columns={
        'pitcher': 'pitcher_id',
        'release_speed': 'avg_release_speed',
        'release_spin_rate': 'avg_spin_rate',
        'pitch_number': 'total_pitches',
        'description': 'total_strikeouts',
        'events': 'total_walks'
    }, inplace=True)

    # Convert to int to avoid float IDs
    summary['pitcher_id'] = summary['pitcher_id'].astype(int)

    # Reverse lookup pitcher names
    try:
        ids = summary['pitcher_id'].tolist()
        lookup = playerid_reverse_lookup(ids)
        lookup['player_name'] = lookup['name_first'].str.strip() + ' ' + lookup['name_last'].str.strip()
        summary = summary.merge(
            lookup[['key_mlbam', 'player_name']],
            how='left',
            left_on='pitcher_id',
            right_on='key_mlbam'
        ).drop(columns=['key_mlbam'])

        # Move name next to ID
        cols = ['pitcher_id', 'player_name'] + [col for col in summary.columns if col not in ['pitcher_id', 'player_name']]
        summary = summary[cols]
    except Exception as e:
        logger.warning(f"Failed to map pitcher names: {e}")

    # Extract game date
    filename = os.path.basename(statcast_path)
    date_part = filename.replace("statcast_", "").replace(".csv", "")

    # Save output
    output_dir = r"C:\Users\roman\baseball_forecast_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pitcher_event_features_{date_part}.csv")
    summary.to_csv(output_path, index=False)
    logger.info(f"Saved pitcher-level features to: {output_path}")

    return output_path

# Optional test run
if __name__ == "__main__":
    raw_dir = r"C:\Users\roman\baseball_forecast_project\data\raw"
    statcast_files = [f for f in os.listdir(raw_dir) if f.startswith("statcast_") and f.endswith(".csv")]

    if not statcast_files:
        logger.error("No Statcast files found in raw directory.")
        sys.exit(1)

    statcast_files.sort(reverse=True)
    latest_file = statcast_files[0]
    full_path = os.path.join(raw_dir, latest_file)

    build_pitcher_event_features(full_path)

    # cd C:\Users\roman\baseball_forecast_project\features
    # python build_pitcher_event_features.py