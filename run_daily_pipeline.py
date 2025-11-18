# run_daily_pipeline.py

import os
import logging
from datetime import datetime, date
from pathlib import Path
import argparse
import pandas as pd  # <-- moved up (used in both branches)

# === Component imports ===
from scraping.scrape_matchups import run_scrape_matchups
from scraping.scrape_statcast import scrape_statcast_today_or_recent
from utils.build_batter_team_lookup import build_batter_team_lookup
from features.build_player_event_features import build_player_event_features
from features.build_pitcher_stat_features import build_pitcher_stat_features
from utils.map_batter_ids import enrich_batter_features_by_team
from features.generate_historical_features import generate_all_historical_features
from features.main_features import build_main_features
from features.historical_main_features import build_historical_main_dataset
from modeling.train_xgb import train_model  # XGBoost version
from modeling.generate_power_rankings import build_rankings  # <-- added

# --- simulation data source ---
from data_source.simulation import SimulationDataSource

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Directory setup ===
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
UI_DATA = BASE_DIR / "ui" / "data"                          # <-- added
UI_DATA.mkdir(parents=True, exist_ok=True)                  # <-- added
WEIGHTS_YML = BASE_DIR / "config" / "ui_weights.yml"        # <-- added


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["realtime", "sim"], default="realtime",
                    help="realtime = scrape & build today; sim = load historical as 'today'")
    ap.add_argument("--game-date", help="YYYY-MM-DD (required for --mode sim)")
    return ap.parse_args()


def run_pipeline():
    args = parse_args()
    logger.info(f"Starting daily MLB prediction pipeline (mode={args.mode})...")

    # Common training file
    historical_path = str(PROCESSED_DIR / "historical_main_features.csv")

    # ---------------------------------------------------------------------
    # SIMULATION MODE: skip scraping; load historical via SimulationDataSource
    # ---------------------------------------------------------------------
    if args.mode == "sim":
        if not args.game_date:
            raise SystemExit("For --mode sim you must pass --game-date YYYY-MM-DD")

        sim_game_date = datetime.strptime(args.game_date, "%Y-%m-%d").date()
        logger.info(f"[SIM] Loading historical artifacts for {sim_game_date} as 'today'...")

        sim = SimulationDataSource(game_date=sim_game_date, redate=True)
        loaded = sim.load()
        logger.info(f"[SIM] Source files used: {loaded.source_files}")

        if loaded.main_features is None:
            raise SystemExit("[SIM] No main_features for that date. "
                             "Either generate main_features_YYYY-MM-DD.csv or add component->builder call here.")

        preds_df = train_model(historical_path=historical_path, today_df=loaded.main_features)
        # preds_df columns: Game Date, Home Team, Away Team, Win Probability, Prediction

        matchup_date = sim_game_date
        statcast_date_str = datetime.today().strftime('%Y-%m-%d')  # keep your "using_{statcast_date}" convention
        output_name = f"readable_win_predictions_for_{matchup_date}_using_{statcast_date_str}.csv"
        readable_path = PREDICTIONS_DIR / output_name
        preds_df.to_csv(readable_path, index=False)
        logger.info(f"[SIM] Saved predictions to: {readable_path}")

        # --- Generate power rankings for UI (SIM mode) ---
        logger.info("[SIM] Generating team power rankings for UI...")
        build_rankings(cfg_path=WEIGHTS_YML, out_dir=UI_DATA)
        logger.info("[SIM] Team power rankings updated.")

        return

    # ---------------------------------------------------------------------
    # REALTIME MODE: your original end-to-end flow (unchanged except small fixes)
    # ---------------------------------------------------------------------

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

    # === Step 3.5: Build batter-to-team lookup file ===
    try:
        logger.info("Step 3.5: Building batter-to-team lookup...")
        build_batter_team_lookup(statcast_file)
        logger.info("Batter-to-team lookup file created.")
    except Exception as e:
        logger.error(f"Failed to build batter-to-team lookup: {e}")
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

    # === Step 6: Build today's main features file ===
    try:
        logger.info("Building main features for today's matchups...")
        main_features_path = build_main_features(matchup_csv_path, pitcher_feature_file, team_feature_file)
        logger.info(f"Main features saved to: {main_features_path}")
    except Exception as e:
        logger.error(f"Failed to build main features: {e}")
        return

    # === Step 6A: Generate historical features before building training dataset ===
    try:
        logger.info("Step 6A: Generating all historical feature files...")
        generate_all_historical_features()
        logger.info("All historical features generated.")
    except Exception as e:
        logger.error(f"Failed to generate historical features: {e}")
        return

    # === Step 6B: Rebuild historical training dataset ===
    try:
        logger.info("Step 6B: Rebuilding historical training dataset...")
        build_historical_main_dataset()
        logger.info("Updated historical_main_features.csv for model training.")
    except Exception as e:
        logger.error(f"Failed to update historical dataset: {e}")
        return

    # === Step 6C: Generate Team Power Rankings for UI ===
    try:
        logger.info("Step 6C: Generating team power rankings for UI...")
        build_rankings(cfg_path=WEIGHTS_YML, out_dir=UI_DATA)
        logger.info("Team power rankings saved to ui/data.")
    except Exception as e:
        logger.warning(f"Power rankings step skipped: {e}")

    # === Step 7: Train model and generate predictions ===
    try:
        predictions_df = train_model(historical_path, main_features_path)  # unchanged API
    except Exception as e:
        logger.error(f"Error during model training or prediction: {e}")
        return

    # === Step 8: Filter predictions for today's matchups ===
    try:
        matchups = pd.read_csv(matchup_csv_path)
        matchups.dropna(subset=["home_team", "away_team"], inplace=True)
        matchups.drop_duplicates(subset=["game_date", "home_team", "away_team"], inplace=True)
        matchups['game_date'] = pd.to_datetime(matchups['game_date'], errors='coerce').dt.date

        today = datetime.today().date()  # realtime only
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
            return translation_dict.get(str(name).strip().upper(), str(name).strip().upper())

        matchups_today['home_team'] = matchups_today['home_team'].astype(str).apply(normalize)
        matchups_today['away_team'] = matchups_today['away_team'].astype(str).apply(normalize)
        matchups_today['matchup_key'] = matchups_today['home_team'] + "_" + matchups_today['away_team']

        predictions_df['matchup_key'] = predictions_df['Home Team'] + "_" + predictions_df['Away Team']
        filtered = predictions_df[predictions_df['matchup_key'].isin(matchups_today['matchup_key'])].copy()
        filtered = filtered.merge(matchups_today[['matchup_key', 'game_date']], on='matchup_key', how='left')
        filtered.drop(columns=['matchup_key'], inplace=True)

        filtered_path = PREDICTIONS_DIR / "today_and_tomorrow_predictions.csv"
        filtered.to_csv(filtered_path, index=False)
        logger.info(f"Filtered predictions saved to: {filtered_path}")
    except Exception as e:
        logger.error(f"Failed to filter predictions: {e}")
        return

    # === Step 9: Format final readable output ===
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

        if isinstance(statcast_actual_date, str):
            statcast_actual_date = datetime.strptime(statcast_actual_date, "%Y-%m-%d").date()

        output_name = f"readable_win_predictions_for_{scraped_game_date_str}_using_{statcast_actual_date.strftime('%Y-%m-%d')}.csv"
        readable_path = PREDICTIONS_DIR / output_name
        readable.to_csv(readable_path, index=False)
        logger.info(f"Clean, deduplicated predictions saved to: {readable_path}")
    except Exception as e:
        logger.error(f"Failed to format readable predictions: {e}")


if __name__ == "__main__":
    run_pipeline()

# cd C:\Users\roman\baseball_forecast_project

#for realtime mode:
# python run_daily_pipeline.py

# For simulation mode:
# python run_daily_pipeline.py --mode sim --game-date 2025-09-20
