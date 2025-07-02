# run_daily_pipeline.py

import os
import logging
from datetime import datetime

# === Component imports ===
from scraping.scrape_matchups import run_scrape_matchups
from scraping.scrape_statcast import scrape_statcast_today_or_recent
from features.build_player_event_features import build_player_event_features
from features.build_pitcher_stat_features import build_pitcher_stat_features
from utils.map_batter_ids import enrich_batter_features_by_team
from modeling.train_model import train_model

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Starting daily MLB prediction pipeline...")

    # === Step 1: Scrape today's matchups ===
    try:
        logger.info("Step 1: Scraping today's MLB matchups...")
        matchup_csv_path, scraped_game_date = run_scrape_matchups()
        scraped_game_date_str = scraped_game_date.strftime('%Y-%m-%d')
        logger.info(f"Matchups scraped and saved to: {matchup_csv_path}")
    except Exception as e:
        logger.error(f"Failed to scrape matchups: {e}")
        return

    # === Step 2: Scrape recent Statcast data ===
    statcast_file, statcast_actual_date = scrape_statcast_today_or_recent(n_days=3)
    if not statcast_file:
        logger.error("No Statcast data found. Exiting pipeline.")
        return
    logger.info(f"Using Statcast data from: {statcast_actual_date}")

    # === Step 3: Player-level features ===
    player_feature_file = build_player_event_features(statcast_file)
    if not player_feature_file:
        logger.error("Failed to build player-level features.")
        return

    # === Step 4: Aggregate batter stats by team ===
    try:
        team_feature_file = enrich_batter_features_by_team(player_feature_file, matchup_csv_path)
        if not team_feature_file:
            logger.error("Failed to build team batter features.")
            return
        logger.info(f"Team batter features saved to: {team_feature_file}")
    except Exception as e:
        logger.error(f"Team aggregation failed: {e}")
        return

    # === Step 5: Aggregate pitcher stats ===
    try:
        pitcher_feature_file = build_pitcher_stat_features(matchup_csv_path)
        if not pitcher_feature_file:
            logger.error("Failed to build pitcher features.")
            return
        logger.info(f"Pitcher stats saved to: {pitcher_feature_file}")
    except Exception as e:
        logger.error(f"Pitcher aggregation failed: {e}")
        return

    # === Step 6: Train model and generate predictions ===
    try:
        historical_path = os.path.join("data", "processed", "historical_main_features.csv")
        today_path = os.path.join("data", "processed", f"main_features_{scraped_game_date_str}.csv")
        predictions_df = train_model(historical_path, today_path)
    except Exception as e:
        logger.error(f"Error during model training or prediction: {e}")
        return

    # === Step 7: Filter predictions for today's matchups ===
    try:
        import pandas as pd

        matchups = pd.read_csv(matchup_csv_path)
        matchups.dropna(subset=["home_team", "away_team"], inplace=True)
        matchups.drop_duplicates(subset=["game_date", "home_team", "away_team"], inplace=True)
        matchups['game_date'] = pd.to_datetime(matchups['game_date'], errors='coerce').dt.date

        today = datetime.today().date()
        matchups_today = matchups[matchups['game_date'] == today].copy()

        if matchups_today.empty:
            logger.warning(f"No matchups found for {today}")
            return

        translation_dict = {
            'RED SOX': 'BOS', 'YANKEES': 'NYY', 'BLUE JAYS': 'TOR', 'ORIOLES': 'BAL', 'RAYS': 'TB',
            'GUARDIANS': 'CLE', 'WHITE SOX': 'CHW', 'ROYALS': 'KC', 'TIGERS': 'DET', 'TWINS': 'MIN',
            'ASTROS': 'HOU', 'MARINERS': 'SEA', 'RANGERS': 'TEX', 'ANGELS': 'LAA', 'ATHLETICS': 'OAK',
            'BRAVES': 'ATL', 'MARLINS': 'MIA', 'METS': 'NYM', 'PHILLIES': 'PHI', 'NATIONALS': 'WSH',
            'BREWERS': 'MIL', 'CARDINALS': 'STL', 'CUBS': 'CHC', 'PIRATES': 'PIT', 'REDS': 'CIN',
            'DODGERS': 'LAD', 'GIANTS': 'SF', 'PADRES': 'SD', 'ROCKIES': 'COL', 'DIAMONDBACKS': 'ARI',
            'D-BACKS': 'ARI', 'ATLÉTICOS': 'OAK', 'AZULEJOS': 'TOR', 'CARDENALES': 'STL',
            'CERVECEROS': 'MIL', 'GIGANTES': 'SF', 'MARINEROS': 'SEA', 'NACIONALES': 'WSH',
            'PIRATAS': 'PIT', 'REALES': 'KC', 'ROJOS': 'CIN', 'TIGRES': 'DET', 'CACHORROS': 'CHC'
        }

        def normalize(name):
            return translation_dict.get(name.strip().upper(), name.strip().upper())

        matchups_today['home_team'] = matchups_today['home_team'].astype(str).apply(normalize)
        matchups_today['away_team'] = matchups_today['away_team'].astype(str).apply(normalize)
        matchups_today['matchup_key'] = matchups_today['home_team'] + "_" + matchups_today['away_team']

        predictions_df['matchup_key'] = predictions_df['Home Team'] + "_" + predictions_df['Away Team']
        filtered = predictions_df[predictions_df['matchup_key'].isin(matchups_today['matchup_key'])].copy()
        filtered = filtered.merge(matchups_today[['matchup_key', 'game_date']], on='matchup_key', how='left')
        filtered.drop(columns=['matchup_key'], inplace=True)

        filtered_path = os.path.join("data", "predictions", "today_and_tomorrow_predictions.csv")
        filtered.to_csv(filtered_path, index=False)
        logger.info(f"Filtered predictions saved to: {filtered_path}")
    except Exception as e:
        logger.error(f"Failed to filter predictions: {e}")
        return

    # === Step 8: Format final readable output ===
    try:
        df = pd.read_csv(filtered_path)

        readable = df[['game_date', 'Home Team', 'Away Team', 'Win Probability']].copy()
        readable['Win Probability'] = readable['Win Probability'].round(2)
        readable['Prediction'] = readable.apply(
            lambda row: f"Pick: {row['Home Team']}" if row['Win Probability'] >= 0.5 else f"Pick: {row['Away Team']}",
            axis=1
        )

        readable = readable.rename(columns={
            'game_date': 'Game Date',
            'Home Team': 'Home Team',
            'Away Team': 'Away Team',
            'Win Probability': 'Win Probability',
            'Prediction': 'Prediction'
        })

        readable.sort_values(by=['Game Date', 'Home Team'], inplace=True)
        readable.drop_duplicates(subset=['Game Date', 'Home Team', 'Away Team'], inplace=True)

        # Convert statcast date to datetime if needed
        if isinstance(statcast_actual_date, str):
            statcast_actual_date = datetime.strptime(statcast_actual_date, "%Y-%m-%d").date()

        output_name = f"readable_win_predictions_for_{scraped_game_date_str}_using_{statcast_actual_date.strftime('%Y-%m-%d')}.csv"
        readable_path = os.path.join("data", "predictions", output_name)
        readable.to_csv(readable_path, index=False)
        logger.info(f"Clean, deduplicated predictions saved to: {readable_path}")
    except Exception as e:
        logger.error(f"Failed to format readable predictions: {e}")

if __name__ == "__main__":
    run_pipeline()


# cd C:\Users\roman\baseball_forecast_project
# python run_daily_pipeline.py