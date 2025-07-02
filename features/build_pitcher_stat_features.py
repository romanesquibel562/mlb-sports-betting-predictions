# build_pitcher_stat_features.py

from pybaseball import statcast, playerid_lookup
from unidecode import unidecode
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_matchup_file(directory: str) -> str:
    files = glob.glob(os.path.join(directory, "mlb_probable_pitchers_*.csv"))
    if not files:
        logger.error("No matchup files found.")
        return None
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"Using matchup file: {latest_file}")
    return latest_file

def extract_game_date_from_filename(filepath: str) -> datetime.date:
    filename = os.path.basename(filepath)
    date_part = filename.replace("mlb_probable_pitchers_", "").replace(".csv", "")
    try:
        return datetime.strptime(date_part, "%Y-%m-%d").date()
    except Exception as e:
        logger.warning(f"Failed to parse date from filename '{filename}': {e}")
        return datetime.today().date()

def build_pitcher_stat_features(matchup_path):
    try:
        logger.info(f"Loading matchup file: {matchup_path}")
        matchups = pd.read_csv(matchup_path)

        # Use matchup file date as Statcast end date
        end_date = extract_game_date_from_filename(matchup_path)
        start_date = end_date - timedelta(days=30)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        pitcher_names = pd.concat([matchups['home_pitcher'], matchups['away_pitcher']]).dropna().unique()
        pitcher_names = [unidecode(name.strip()) for name in pitcher_names]
        logger.info(f"Found {len(pitcher_names)} unique pitchers: {pitcher_names}")

        pitcher_records = []
        for full_name in pitcher_names:
            if " " not in full_name:
                logger.warning(f"Invalid pitcher name format: {full_name}")
                continue
            first, last = full_name.split(" ", 1)
            lookup = playerid_lookup(last.title(), first.title())
            if lookup.empty or pd.isna(lookup['key_mlbam'].values[0]):
                logger.warning(f"MLBAM ID not found for: {full_name}")
                continue
            pitcher_records.append({'full_name': full_name, 'mlbam_id': int(lookup['key_mlbam'].values[0])})

        pitcher_df = pd.DataFrame(pitcher_records)
        if pitcher_df.empty:
            logger.warning("No valid pitchers with MLBAM IDs.")
            return None

        logger.info(f"Downloading Statcast data from {start_str} to {end_str}...")
        statcast_df = statcast(start_dt=start_str, end_dt=end_str)
        logger.info(f"Total Statcast rows pulled: {len(statcast_df)}")

        filtered_df = statcast_df[statcast_df['pitcher'].isin(pitcher_df['mlbam_id'])].copy()
        logger.info(f"Filtered to {len(filtered_df)} rows for target pitchers")

        filtered_df = filtered_df.merge(pitcher_df, how='left', left_on='pitcher', right_on='mlbam_id')

        metrics_df = (
            filtered_df.groupby('pitcher')
            .agg(
                full_name=('full_name', 'first'),
                total_pitches=('pitch_type', 'count'),
                avg_velocity=('release_speed', 'mean'),
                avg_spin_rate=('release_spin_rate', 'mean'),
                avg_extension=('release_extension', 'mean'),
                strikeouts=('events', lambda x: (x == 'strikeout').sum()),
                whiffs=('events', lambda x: (x == 'swinging_strike').sum()),
                avg_bat_speed=('bat_speed', 'mean'),
                avg_launch_angle=('launch_angle', 'mean'),
                avg_exit_velocity=('launch_speed', 'mean'),
                avg_swing_length=('swing_length', 'mean'),
                games_played=('game_date', 'nunique')
            )
            .reset_index(drop=True)
        )

        metrics_df = metrics_df.round(2)
        logger.info(f"Final DataFrame built for {len(metrics_df)} pitchers.")

        output_path = os.path.join(
            "C:/Users/roman/baseball_forecast_project/data/processed",
            f"pitcher_stat_features_{end_str}.csv"
        )
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Saved pitcher stat features to: {output_path}")
        print("\nFinal Output:\n", metrics_df.to_string(index=False))
        return output_path

    except Exception as e:
        logger.error(f"Top-level failure: {e}")
        return None

if __name__ == "__main__":
    matchup_dir = "C:/Users/roman/baseball_forecast_project/data/raw"
    latest_file = find_latest_matchup_file(matchup_dir)
    if latest_file:
        result_path = build_pitcher_stat_features(latest_file)
        if result_path:
            df = pd.read_csv(result_path)
            print("\nReloaded CSV:\n", df.to_string(index=False))


# cd C:\Users\roman\baseball_forecast_project\features
        # python build_pitcher_stat_features.py
        