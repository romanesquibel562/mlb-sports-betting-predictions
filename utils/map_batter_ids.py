# map_batter_ids.py

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# === Logger setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Base Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
LOOKUP_PATH = BASE_DIR / "utils" / "data" / "reference" / "batter_team_lookup.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def enrich_batter_features_by_team(player_feature_path: Path,
                                   matchup_path: Path,
                                   batter_lookup_path: Path = None) -> Path | None:
    try:
        # === 1. Load player features ===
        player_df = pd.read_csv(player_feature_path)
        logger.info(f"Loaded player features: {len(player_df)} rows")

        # === 2. Load batter-to-team mapping ===
        lookup_path = batter_lookup_path or LOOKUP_PATH
        lookup_df = pd.read_csv(lookup_path)
        logger.info(f"Loaded batter-team lookup: {len(lookup_df)} rows from {lookup_path}")

        # === 3. Merge player stats with team info ===
        player_df.rename(columns={"mlbam_id": "batter"}, inplace=True)
        merged = pd.merge(player_df, lookup_df, on="batter", how="left")
        if merged['team_name'].isnull().any():
            logger.warning("Some batters could not be matched to a team!")

        # === 4. Load and normalize matchups ===
        matchups = pd.read_csv(matchup_path)
        translation_dict = {
            'RED SOX': 'BOS', 'YANKEES': 'NYY', 'BLUE JAYS': 'TOR', 'ORIOLES': 'BAL', 'RAYS': 'TB',
            'GUARDIANS': 'CLE', 'WHITE SOX': 'CWS', 'ROYALS': 'KC', 'TIGERS': 'DET', 'TWINS': 'MIN',
            'ASTROS': 'HOU', 'MARINERS': 'SEA', 'RANGERS': 'TEX', 'ANGELS': 'LAA', 'ATHLETICS': 'OAK',
            'BRAVES': 'ATL', 'MARLINS': 'MIA', 'METS': 'NYM', 'PHILLIES': 'PHI', 'NATIONALS': 'WSH',
            'BREWERS': 'MIL', 'CARDINALS': 'STL', 'CUBS': 'CHC', 'PIRATES': 'PIT', 'REDS': 'CIN',
            'DODGERS': 'LAD', 'GIANTS': 'SF', 'PADRES': 'SD', 'ROCKIES': 'COL', 'DIAMONDBACKS': 'ARI',
            'ATLÉTICOS': 'OAK', 'AZULEJOS': 'TOR', 'BRAVOS': 'ATL', 'CARDENALES': 'STL',
            'CERVECEROS': 'MIL', 'GIGANTES': 'SF', 'MARINEROS': 'SEA', 'NACIONALES': 'WSH',
            'PIRATAS': 'PIT', 'REALES': 'KC', 'ROJOS': 'CIN', 'TIGRES': 'DET'
        }

        def safe_map(team_series):
            return team_series.apply(lambda x: translation_dict.get(str(x).upper(), str(x).upper()))

        matchups['home_team'] = safe_map(matchups['home_team'])
        matchups['away_team'] = safe_map(matchups['away_team'])

        logger.info(f"Raw home teams: {matchups['home_team'].unique()}")
        logger.info(f"Raw away teams: {matchups['away_team'].unique()}")

        # === 5. Filter players by today's teams ===
        merged['team_name'] = merged['team_name'].str.upper()
        today_teams = set(matchups['home_team']).union(set(matchups['away_team']))
        logger.info(f"Today's teams from matchups: {today_teams}")
        logger.info(f"Team names in merged data: {set(merged['team_name'].unique())}")
        logger.info(f"Overlap: {set(merged['team_name'].unique()) & today_teams}")

        filtered = merged[merged['team_name'].isin(today_teams)].copy()
        if filtered.empty:
            logger.warning("No matching players found for today's matchups.")
            return None

        # === 6. Aggregate batter stats by team ===
        numeric_cols = [
            'avg_launch_speed', 'avg_launch_angle', 'avg_bat_speed',
            'avg_swing_length', 'total_home_runs', 'total_strikeouts',
            'avg_plate_x', 'avg_plate_z'
        ]
        available_cols = [col for col in numeric_cols if col in filtered.columns]

        team_summary = filtered.groupby('team_name')[available_cols].agg('mean').reset_index()

        # === 7. Save output ===
        output_path = PROCESSED_DIR / f"team_batter_stats_{datetime.today().strftime('%Y-%m-%d')}.csv"
        team_summary.to_csv(output_path, index=False)
        logger.info(f"Saved aggregated team batter stats to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to enrich batter features by team: {e}")
        return None

# === Manual test ===
if __name__ == "__main__":
    # Replace with actual filenames as needed for testing
    sample_player_file = PROCESSED_DIR / "player_features_2025-06-27.csv"
    sample_matchup_file = RAW_DIR / "mlb_probable_pitchers_2025-06-28.csv"
    enrich_batter_features_by_team(sample_player_file, sample_matchup_file)

    # cd C:\Users\roman\baseball_forecast_project\utils
    # python map_batter_ids.py
