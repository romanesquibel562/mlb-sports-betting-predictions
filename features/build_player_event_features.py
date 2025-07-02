# build_player_event_features.py

import pandas as pd
import os
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def extract_date_from_filename(filename):
    try:
        return filename.replace("statcast_", "").replace(".csv", "")
    except:
        return None

def build_player_event_features(statcast_path, filter_to_date=None):
    logger.info(f"Building player-level features from: {statcast_path}")

    try:
        df = pd.read_csv(statcast_path)
        logger.info(f"Loaded {len(df)} rows from Statcast.")
    except Exception as e:
        logger.error(f"Failed to load Statcast file: {e}")
        return None

    if df.empty:
        logger.warning("Empty Statcast dataset, skipping player feature generation.")
        return None

    # Optional date filter
    if filter_to_date and 'game_date' in df.columns:
        try:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            target_date = pd.to_datetime(filter_to_date)
            df = df[df['game_date'] == target_date]
            logger.info(f"Filtered Statcast to {len(df)} rows for game date: {filter_to_date}")
        except Exception as e:
            logger.warning(f"Failed to filter by game_date: {e}")

    df['is_home_run'] = df['events'].apply(lambda x: 1 if x == 'home_run' else 0)
    df['is_strike_out'] = df['events'].apply(lambda x: 1 if x == 'strikeout' else 0)
    df['plate_appearance'] = 1

    for col in ['launch_speed', 'bat_speed', 'swing_length', 'plate_x', 'plate_z']:
        if col not in df.columns:
            df[col] = None
            logger.warning(f"Missing column '{col}', filling with NaN.")

    summary = df.groupby('batter').agg({
        'launch_speed': 'mean',
        'bat_speed': 'mean',
        'swing_length': 'mean',
        'is_home_run': 'sum',
        'is_strike_out': 'sum',
        'plate_appearance': 'sum'
    }).reset_index()

    summary['home_run_rate'] = summary['is_home_run'] / summary['plate_appearance']
    summary['strikeout_rate'] = summary['is_strike_out'] / summary['plate_appearance']

    summary.rename(columns={
        'batter': 'mlbam_id',
        'launch_speed': 'avg_launch_speed',
        'bat_speed': 'avg_bat_speed',
        'swing_length': 'avg_swing_length',
        'is_home_run': 'recent_home_runs',
        'is_strike_out': 'recent_strikeouts',
        'plate_appearance': 'plate_appearances'
    }, inplace=True)

    # Merge with lookup
    try:
        lookup_path = r"C:\Users\roman\baseball_forecast_project\evaluation\data\processed\batter_team_lookup.csv"
        lookup_df = pd.read_csv(lookup_path)
        summary = pd.merge(summary, lookup_df, how='left', on='mlbam_id')
        logger.info("Merged batter lookup to add player_name and team.")
    except Exception as e:
        logger.warning(f"Failed to merge batter lookup: {e}")

    date_part = extract_date_from_filename(os.path.basename(statcast_path))
    output_dir = r"C:\Users\roman\baseball_forecast_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"player_features_{date_part}.csv")
    summary.to_csv(output_path, index=False)
    logger.info(f"Saved player-level features to: {output_path}")

    return output_path

if __name__ == "__main__":
    raw_dir = r"C:\Users\roman\baseball_forecast_project\data\raw"
    statcast_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("statcast_")])
    if not statcast_files:
        logger.error("No Statcast files found.")
    else:
        latest_file = statcast_files[-1]
        statcast_path = os.path.join(raw_dir, latest_file)
        filter_date = extract_date_from_filename(latest_file)
        logger.info(f"Running test on latest file: {statcast_path}")
        build_player_event_features(statcast_path, filter_to_date=filter_date)

    # cd C:\Users\roman\baseball_forecast_project\features
    # python build_player_event_features.py