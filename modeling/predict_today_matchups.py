# modeling/predict_today_matchups.py

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_file(folder_path: Path, pattern: str):
    files = sorted(folder_path.glob(pattern), reverse=True)
    return files[0] if files else None

def predict_today_matchups(model, feature_columns, team_feature_path, matchup_path, game_feature_path, output_dir=Path("data/predictions")):
    # Load data
    try:
        team_df = pd.read_csv(team_feature_path)
        matchups = pd.read_csv(matchup_path)
        game_df = pd.read_csv(game_feature_path)
        logger.info(f"Loaded {len(team_df)} teams, {len(matchups)} matchups, and {len(game_df)} game features.")
    except Exception as e:
        logger.error(f"Failed to load required files: {e}")
        return None

    # Normalize team names
    translation_dict = {
        'RED SOX': 'BOS', 'YANKEES': 'NYY', 'BLUE JAYS': 'TOR', 'ORIOLES': 'BAL', 'RAYS': 'TB',
        'GUARDIANS': 'CLE', 'WHITE SOX': 'CHW', 'ROYALS': 'KC', 'TIGERS': 'DET', 'TWINS': 'MIN',
        'ASTROS': 'HOU', 'MARINERS': 'SEA', 'RANGERS': 'TEX', 'ANGELS': 'LAA', 'ATHLETICS': 'OAK',
        'BRAVES': 'ATL', 'MARLINS': 'MIA', 'METS': 'NYM', 'PHILLIES': 'PHI', 'NATIONALS': 'WSH',
        'BREWERS': 'MIL', 'CARDINALS': 'STL', 'CUBS': 'CHC', 'PIRATES': 'PIT', 'REDS': 'CIN',
        'DODGERS': 'LAD', 'GIANTS': 'SF', 'PADRES': 'SD', 'ROCKIES': 'COL', 'DIAMONDBACKS': 'ARI',
        'ATLÉTICOS': 'OAK', 'AZULEJOS': 'TOR', 'BRAVOS': 'ATL', 'CARDENALES': 'STL',
        'CERVECEROS': 'MIL', 'GIGANTES': 'SF', 'MARINEROS': 'SEA', 'NACIONALES': 'WSH',
        'PIRATAS': 'PIT', 'REALES': 'KC', 'ROJOS': 'CIN', 'TIGRES': 'DET', 'CACHORROS': 'CHC',
        'D-BACKS': 'ARI'
    }

    try:
        matchups['home_team'] = matchups['home_team'].str.upper().map(translation_dict).fillna(matchups['home_team'])
        matchups['away_team'] = matchups['away_team'].str.upper().map(translation_dict).fillna(matchups['away_team'])
        logger.info("Normalized team names using translation dictionary.")
    except Exception as e:
        logger.error(f"Failed during team name normalization: {e}")
        return None

    # Merge game-level features
    try:
        enriched = pd.merge(matchups, game_df, on=["home_team", "away_team"], how="left")
        logger.info("Merged game-level features into matchups.")
    except Exception as e:
        logger.error(f"Failed to merge game-level features: {e}")
        return None

    # Merge team-level features
    try:
        team_home = team_df.add_prefix("home_")
        team_home.rename(columns={"home_team_name": "home_team"}, inplace=True)
        team_away = team_df.add_prefix("away_")
        team_away.rename(columns={"away_team_name": "away_team"}, inplace=True)
        enriched = pd.merge(enriched, team_home, on="home_team", how="left")
        enriched = pd.merge(enriched, team_away, on="away_team", how="left")
        logger.info("Merged team features into matchups.")
    except Exception as e:
        logger.error(f"Failed during merging of team stats: {e}")
        return None

    # Handle missing team info
    try:
        numeric_cols = enriched.select_dtypes(include='number').columns
        enriched[numeric_cols] = enriched[numeric_cols].fillna(enriched[numeric_cols].mean())
        logger.info("Filled missing numeric features with league averages.")
    except Exception as e:
        logger.warning(f"Could not fill missing team stats: {e}")

    # Build feature matrix
    try:
        feature_columns = [col for col in feature_columns if col in enriched.columns]
        missing_cols = [col for col in feature_columns if col not in enriched.columns]
        if missing_cols:
            logger.warning(f"The following model features were missing in live data and skipped: {missing_cols}")
        X_today = enriched[feature_columns].fillna(0)
        logger.info(f"Prepared input feature matrix for {len(X_today)} matchups.")
    except Exception as e:
        logger.error(f"Failed to build feature matrix for prediction: {e}")
        return None

    # Predict probabilities
    try:
        enriched['predicted_win_prob'] = model.predict_proba(X_today)[:, 1]
        enriched['win_pick'] = enriched.apply(
            lambda row: f"Pick: {row['home_team']}" if row['predicted_win_prob'] >= 0.5 else f"Pick: {row['away_team']}",
            axis=1
        )
        enriched['game_date'] = matchups['game_date'] if 'game_date' in matchups.columns else datetime.today().date()
        logger.info("Successfully generated predictions.")
    except Exception as e:
        logger.error(f"Prediction step failed: {e}")
        return None

    # Format and save output
    try:
        output = enriched[['game_date', 'home_team', 'away_team', 'predicted_win_prob', 'win_pick']].copy()
        output.rename(columns={
            'game_date': 'Game Date',
            'home_team': 'Home Team',
            'away_team': 'Away Team',
            'predicted_win_prob': 'Win Probability',
            'win_pick': 'Prediction'
        }, inplace=True)
        output['Win Probability'] = output['Win Probability'].round(2)
        output.sort_values(by='Game Date', inplace=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.today().strftime('%Y-%m-%d')
        output_path = output_dir / f"readable_win_predictions_for_{today_str}.csv"
        output.to_csv(output_path, index=False)
        logger.info(f"Saved real-time predictions to: {output_path}")
    except Exception as e:
        logger.error(f"Failed during output formatting/saving: {e}")
        return None

    return output_path

# Standalone test block
if __name__ == "__main__":
    from train_model import train_model

    today = datetime.today().date()
    today_str = today.strftime('%Y-%m-%d')

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"

    team_feature_path = processed_dir / f"team_batter_stats_{today_str}.csv"
    matchup_path = raw_dir / f"mlb_probable_pitchers_{today_str}.csv"
    game_feature_path = processed_dir / f"features_{today_str}.csv"

    if not team_feature_path.exists():
        logger.warning("No team stats for today. Trying fallback...")
        team_feature_path = find_latest_file(processed_dir, "team_batter_stats_2025-*.csv")

    if not matchup_path.exists():
        logger.warning("No matchup file for today. Trying fallback...")
        matchup_path = find_latest_file(raw_dir, "mlb_probable_pitchers_2025-*.csv")

    if not game_feature_path.exists():
        logger.warning("No features file for today. Trying fallback...")
        game_feature_path = find_latest_file(processed_dir, "features_2025-*.csv")

    if not team_feature_path or not matchup_path or not game_feature_path:
        logger.error("Missing one or more required files. Exiting.")
        exit()

    try:
        model, feature_columns, _ = train_model(game_feature_path, team_feature_path, today)
        output_path = predict_today_matchups(model, feature_columns, team_feature_path, matchup_path, game_feature_path)

        if output_path:
            logger.info(f"Real-time prediction pipeline completed successfully. Output saved to: {output_path}")
        else:
            logger.error("Prediction failed. No output file created.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        exit()

    # cd C:\Users\roman\baseball_forecast_project\modeling
    # python predict_today_matchups.py
