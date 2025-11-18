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

        # Normalize headers & keys
        lookup_df.columns = [c.strip() for c in lookup_df.columns]
        if 'mlbam_id' not in lookup_df.columns and 'batter' in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={'batter': 'mlbam_id'})
        if 'mlbam_id' not in player_df.columns:
            raise ValueError("player_features is missing 'mlbam_id' column.")
        player_df['mlbam_id'] = pd.to_numeric(player_df['mlbam_id'], errors='coerce').astype('Int64')
        lookup_df['mlbam_id'] = pd.to_numeric(lookup_df['mlbam_id'], errors='coerce').astype('Int64')

        # === 3. Merge player stats with team info (on mlbam_id) ===
        merged = pd.merge(player_df, lookup_df, on="mlbam_id", how="left")

        # --- NEW: coalesce any team columns into a single 'team_name' ---
        def coalesce_team(df: pd.DataFrame) -> pd.DataFrame:
            candidates = [
                'team_name', 'team', 'team_abbr',
                'team_name_x', 'team_name_y',
                'team_x', 'team_y',
                'team_abbr_x', 'team_abbr_y'
            ]
            found = [c for c in candidates if c in df.columns]
            if not found:
                raise KeyError(f"No team column found after merge. Columns: {list(df.columns)}")
            # build 'team_name' from first non-null across candidates
            out = df.get(found[0]).copy()
            for c in found[1:]:
                out = out.fillna(df[c])
            df['team_name'] = out
            return df

        merged = coalesce_team(merged)

        if merged['team_name'].isnull().any():
            logger.warning("Some batters could not be matched to a team!")

        # === 4. Load and normalize matchups (long → abbr) ===
        matchups = pd.read_csv(matchup_path)
        LONG_TO_ABBR = {
            'ARIZONA DIAMONDBACKS':'ARI','ATLANTA BRAVES':'ATL','BALTIMORE ORIOLES':'BAL','BOSTON RED SOX':'BOS',
            'CHICAGO CUBS':'CHC','CHICAGO WHITE SOX':'CWS','CINCINNATI REDS':'CIN','CLEVELAND GUARDIANS':'CLE',
            'COLORADO ROCKIES':'COL','DETROIT TIGERS':'DET','HOUSTON ASTROS':'HOU','KANSAS CITY ROYALS':'KC',
            'LOS ANGELES ANGELS':'LAA','LOS ANGELES DODGERS':'LAD','MIAMI MARLINS':'MIA','MILWAUKEE BREWERS':'MIL',
            'MINNESOTA TWINS':'MIN','NEW YORK METS':'NYM','NEW YORK YANKEES':'NYY','OAKLAND ATHLETICS':'OAK',
            'PHILADELPHIA PHILLIES':'PHI','PITTSBURGH PIRATES':'PIT','SAN DIEGO PADRES':'SD','SEATTLE MARINERS':'SEA',
            'SAN FRANCISCO GIANTS':'SF','ST. LOUIS CARDINALS':'STL','ST LOUIS CARDINALS':'STL','TAMPA BAY RAYS':'TB',
            'TEXAS RANGERS':'TEX','TORONTO BLUE JAYS':'TOR','WASHINGTON NATIONALS':'WSH'
        }
        ABBR_ALIASES = {'CHW':'CWS','WAS':'WSH','WSN':'WSH','TBR':'TB','KCR':'KC','SDP':'SD','SFG':'SF'}
        VALID_ABBRS = set(LONG_TO_ABBR.values())

        def to_abbr(x):
            s = str(x).strip().upper()
            s = ABBR_ALIASES.get(s, s)
            return s if s in VALID_ABBRS else LONG_TO_ABBR.get(s, s)

        matchups['home_team'] = matchups['home_team'].map(to_abbr)
        matchups['away_team'] = matchups['away_team'].map(to_abbr)

        logger.info(f"Raw home teams: {matchups['home_team'].unique()}")
        logger.info(f"Raw away teams: {matchups['away_team'].unique()}")

        # === 5. Filter players by today's teams (abbr on both sides) ===
        merged['team_name'] = merged['team_name'].str.upper()
        today_teams = set(matchups['home_team']) | set(matchups['away_team'])
        logger.info(f"Today's teams from matchups: {today_teams}")
        logger.info(f"Team names in merged data: {set(merged['team_name'].unique())}")
        logger.info(f"Overlap: {set(merged['team_name'].unique()) & today_teams}")

        filtered = merged.dropna(subset=['team_name'])
        filtered = filtered[filtered['team_name'].isin(today_teams)].copy()
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
    sample_player_file = PROCESSED_DIR / "player_features_2025-10-08.csv"
    sample_matchup_file = RAW_DIR / "mlb_probable_pitchers_2025-10-08.csv"
    enrich_batter_features_by_team(sample_player_file, sample_matchup_file)

    # cd C:\Users\roman\baseball_forecast_project\utils
    # python map_batter_ids.py
