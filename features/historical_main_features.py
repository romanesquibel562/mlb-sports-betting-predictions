# historical_main_features.py

import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === PATH SETUP ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
HISTORICAL_MATCHUP_DIR = RAW_DIR / "historical_matchups"

RAW_OUTPUT_PATH = PROCESSED_DIR / "historical_main_features_raw.csv"
CLEAN_OUTPUT_PATH = PROCESSED_DIR / "historical_main_features.csv"

# === TEAM ABBREVIATION MAP ===
TEAM_ABBREV_MAP = {
    "RED SOX": "BOS", "YANKEES": "NYY", "BLUE JAYS": "TOR", "RAYS": "TBR", "ORIOLES": "BAL",
    "WHITE SOX": "CHW", "GUARDIANS": "CLE", "TIGERS": "DET", "ROYALS": "KCR", "TWINS": "MIN",
    "ASTROS": "HOU", "MARINERS": "SEA", "RANGERS": "TEX", "ATHLETICS": "OAK", "ANGELS": "LAA",
    "BRAVES": "ATL", "PHILLIES": "PHI", "METS": "NYM", "MARLINS": "MIA", "NATIONALS": "WSH",
    "CUBS": "CHC", "REDS": "CIN", "PIRATES": "PIT", "BREWERS": "MIL", "CARDINALS": "STL",
    "DODGERS": "LAD", "GIANTS": "SFG", "ROCKIES": "COL", "PADRES": "SDP", "DIAMONDBACKS": "ARI",
    "D-BACKS": "ARI"
}


def normalize_name(name):
    if pd.isna(name):
        return ""
    return (
        name.upper().strip()
            .replace("Á", "A").replace("É", "E").replace("Í", "I")
            .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
            .replace(".", "")
    )


def extract_date_from_filename(path: Path, prefix: str):
    return path.name.replace(f"{prefix}_", "").replace(".csv", "")


def load_csv_by_prefix(prefix, date_str, directory=PROCESSED_DIR):
    path = directory / f"{prefix}_{date_str}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"{prefix} for {date_str} is empty.")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to load {prefix} for {date_str}: {e}")
            return None
    else:
        logger.warning(f"Missing {prefix} for {date_str}")
        return None


def map_abbrev(team):
    return TEAM_ABBREV_MAP.get(team.upper().strip(), team.upper().strip())


def build_historical_main_dataset():
    result_files = sorted(PROCESSED_DIR.glob("historical_results_*.csv"))
    if not result_files:
        logger.error("No historical_results_*.csv files found.")
        return

    all_rows = []

    for result_file in result_files:
        date_str = extract_date_from_filename(result_file, "historical_results")
        logger.info(f"Processing game date: {date_str}")

        try:
            results_df = pd.read_csv(result_file)
            matchup_df = load_csv_by_prefix("historical_matchups", date_str, HISTORICAL_MATCHUP_DIR)
            pitcher_df = load_csv_by_prefix("pitcher_stat_features", date_str)
            batter_df = load_csv_by_prefix("batter_stat_features", date_str)
            team_df = load_csv_by_prefix("team_form", date_str)
        except Exception as e:
            logger.error(f"Error reading CSVs for {date_str}: {e}")
            continue

        if any(x is None for x in [matchup_df, pitcher_df, batter_df, team_df]):
            logger.warning(f"Skipping {date_str} due to missing or invalid files.")
            continue

        try:
            # Normalize team and player names
            matchup_df["home_pitcher"] = matchup_df["home_pitcher"].apply(normalize_name)
            matchup_df["away_pitcher"] = matchup_df["away_pitcher"].apply(normalize_name)
            matchup_df["home_team"] = matchup_df["home_team"].str.upper().str.strip()
            matchup_df["away_team"] = matchup_df["away_team"].str.upper().str.strip()
            pitcher_df["full_name"] = pitcher_df["full_name"].apply(normalize_name)

            # Merge pitcher stats
            df = matchup_df.merge(pitcher_df, left_on="home_pitcher", right_on="full_name", how="left")
            df = df.rename(columns={col: f"home_pitcher_{col}" for col in pitcher_df.columns if col != "full_name"})
            df.drop(columns=["full_name"], inplace=True)

            df = df.merge(pitcher_df, left_on="away_pitcher", right_on="full_name", how="left")
            df = df.rename(columns={col: f"away_pitcher_{col}" for col in pitcher_df.columns if col != "full_name"})
            df.drop(columns=["full_name"], inplace=True)

            # Merge batter stats
            batter_df["team"] = batter_df["team"].str.upper().str.strip()
            df = df.merge(batter_df, left_on="home_team", right_on="team", how="left")
            df = df.rename(columns={col: f"home_team_{col}" for col in batter_df.columns if col != "team"})
            df.drop(columns=["team"], inplace=True)
            df = df.merge(batter_df, left_on="away_team", right_on="team", how="left")
            df = df.rename(columns={col: f"away_team_{col}" for col in batter_df.columns if col != "team"})
            df.drop(columns=["team"], inplace=True)

            # Merge team form
            team_df["team"] = team_df["team"].str.upper().str.strip()
            df = df.merge(team_df.add_prefix("home_"), left_on="home_team", right_on="home_team", how="left")
            df = df.merge(team_df.add_prefix("away_"), left_on="away_team", right_on="away_team", how="left")

            # Normalize and merge actual results
            results_df["home_team"] = results_df["home_team"].str.upper().str.strip()
            results_df["away_team"] = results_df["away_team"].str.upper().str.strip()
            results_df["winner"] = results_df["winner"].str.upper().str.strip()

            df = df.merge(results_df, on=["game_date", "home_team", "away_team"], how="inner")
            df["actual_winner"] = df["winner"]

            all_rows.append(df)

        except Exception as e:
            logger.error(f"Merge failed for {date_str}: {e}")
            continue

    if not all_rows:
        logger.warning("No rows processed. Final dataset was not created.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(RAW_OUTPUT_PATH, index=False)

    # CLEANING
    final_df.drop_duplicates(subset=["game_date", "home_team", "away_team"], inplace=True)
    final_df.dropna(axis=1, how="all", inplace=True)

    for col in final_df.select_dtypes(include=["float64", "int64"]).columns:
        final_df[col] = final_df[col].fillna(final_df[col].mean())

    final_df = final_df[final_df["actual_winner"].notna()]
    final_df["home_team"] = final_df["home_team"].map(map_abbrev)
    final_df["away_team"] = final_df["away_team"].map(map_abbrev)
    final_df["actual_winner"] = final_df["actual_winner"].map(map_abbrev)

    final_df.to_csv(CLEAN_OUTPUT_PATH, index=False)
    logger.info(f"Saved clean final dataset to {CLEAN_OUTPUT_PATH} with {len(final_df)} rows.")


if __name__ == "__main__":
    build_historical_main_dataset()

# Run command from terminal:
# cd C:\Users\roman\baseball_forecast_project\features
# python historical_main_features.py
