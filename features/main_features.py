# main_features.py

import pandas as pd
import os
import logging
from datetime import datetime

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def normalize_name(name):
    if pd.isna(name):
        return ""
    return (
        name.upper()
            .strip()
            .replace("Á", "A")
            .replace("É", "E")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ú", "U")
            .replace("Ñ", "N")
            .replace(".", "")
    )

def main():
    today_str = datetime.today().strftime("%Y-%m-%d")

    # File paths
    matchup_path = f"C:/Users/roman/baseball_forecast_project/data/raw/mlb_probable_pitchers_{today_str}.csv"
    pitcher_path = f"C:/Users/roman/baseball_forecast_project/data/processed/pitcher_stat_features_{today_str}.csv"
    batter_path = f"C:/Users/roman/baseball_forecast_project/data/processed/team_batter_stats_{today_str}.csv"
    team_form_path = "C:/Users/roman/baseball_forecast_project/data/processed/team_recent_form.csv"
    output_path = f"C:/Users/roman/baseball_forecast_project/data/processed/main_features_{today_str}.csv"

    # Load matchups
    matchups = pd.read_csv(matchup_path)
    matchups["home_pitcher"] = matchups["home_pitcher"].apply(normalize_name)
    matchups["away_pitcher"] = matchups["away_pitcher"].apply(normalize_name)
    matchups["home_team"] = matchups["home_team"].str.upper().str.strip()
    matchups["away_team"] = matchups["away_team"].str.upper().str.strip()
    logger.info(f"Loaded matchups: {len(matchups)} rows from {matchup_path}")

    # Load pitcher stats
    pitchers = pd.read_csv(pitcher_path)
    pitchers["full_name"] = pitchers["full_name"].apply(normalize_name)
    logger.info(f"Loaded pitcher stats: {len(pitchers)} rows from {pitcher_path}")

    # Merge home pitcher stats
    df = matchups.merge(pitchers, left_on="home_pitcher", right_on="full_name", how="left")
    df = df.rename(columns={col: f"home_pitcher_{col}" for col in pitchers.columns if col != "full_name"})
    df.drop(columns=["full_name"], inplace=True)

    # Merge away pitcher stats
    df = df.merge(pitchers, left_on="away_pitcher", right_on="full_name", how="left")
    df = df.rename(columns={col: f"away_pitcher_{col}" for col in pitchers.columns if col != "full_name"})
    df.drop(columns=["full_name"], inplace=True)

    # Log unmatched pitchers
    unmatched_home = df[df["home_pitcher_avg_velocity"].isna()]["home_pitcher"].unique()
    unmatched_away = df[df["away_pitcher_avg_velocity"].isna()]["away_pitcher"].unique()
    if len(unmatched_home) > 0:
        logger.warning("Missing home pitcher stats for:")
        for name in unmatched_home:
            logger.warning(f" - {repr(name)}")
    if len(unmatched_away) > 0:
        logger.warning("Missing away pitcher stats for:")
        for name in unmatched_away:
            logger.warning(f" - {repr(name)}")

    # Load team batter stats
    batter_stats = pd.read_csv(batter_path)
    batter_stats["team_name"] = batter_stats["team_name"].str.upper().str.strip()
    logger.info(f"Loaded team batter stats: {len(batter_stats)} rows from {batter_path}")

    df = df.merge(batter_stats, left_on="home_team", right_on="team_name", how="left")
    df = df.rename(columns={col: f"home_team_{col}" for col in batter_stats.columns if col != "team_name"})
    df.drop(columns=["team_name"], inplace=True)

    df = df.merge(batter_stats, left_on="away_team", right_on="team_name", how="left")
    df = df.rename(columns={col: f"away_team_{col}" for col in batter_stats.columns if col != "team_name"})
    df.drop(columns=["team_name"], inplace=True)

    # Log unmatched batter stats
    if df[["home_team_avg_launch_speed", "away_team_avg_launch_speed"]].isna().any().any():
        logger.warning("Some home or away team batter stats are missing.")

    # Load team recent form
    form = pd.read_csv(team_form_path)
    form["team"] = form["team"].str.upper().str.strip()
    logger.info(f"Loaded team form: {len(form)} rows from {team_form_path}")

    df = df.merge(form.add_prefix("home_"), left_on="home_team", right_on="home_team", how="left")
    df = df.merge(form.add_prefix("away_"), left_on="away_team", right_on="away_team", how="left")

    # Save final features
    df.to_csv(output_path, index=False)
    logger.info(f"Saved main features to: {output_path}")

if __name__ == "__main__":
    main()


    # cd C:\Users\roman\baseball_forecast_project\features
    # python main_features.py