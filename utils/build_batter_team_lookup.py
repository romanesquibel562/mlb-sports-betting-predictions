# build_batter_team_lookup.py

import pandas as pd
import os
from datetime import datetime
import logging
from pybaseball import playerid_reverse_lookup

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        # Save to correct reference path
        output_path = os.path.join(
            "C:/Users/roman/baseball_forecast_project/utils/data/reference",
            "batter_team_lookup.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)

        logger.info(f"Saved batter-team lookup to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to build batter-team lookup: {e}")
        return None

# === Run standalone ===
if __name__ == "__main__":
    latest_statcast = r"C:\Users\roman\baseball_forecast_project\data\raw\statcast_2025-06-27.csv"
    build_batter_team_lookup(latest_statcast)

# cd C:\Users\roman\baseball_forecast_project\utils
# python build_batter_team_lookup.py

