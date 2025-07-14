# train_model.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Root directory setup ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def train_model(historical_path, today_path):
    logger.info(f"Loading historical dataset from {historical_path}")
    historical_df = pd.read_csv(historical_path)

    # Ensure only numeric columns are used for training
    non_feature_cols = [
        "actual_winner", "game_date", "home_team", "away_team",
        "home_pitcher", "away_pitcher", "home_pitcher_full_name", "away_pitcher_full_name"
    ]
    numeric_cols = [col for col in historical_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(historical_df[col])]

    X_train = historical_df[numeric_cols]
    y_train = (historical_df["actual_winner"] == historical_df["home_team"]).astype(int)

    logger.info(f"Training on {len(X_train)} historical games with {X_train.shape[1]} features")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1]

    acc = accuracy_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mape = np.mean(np.abs((y_train - y_pred) / np.maximum(np.abs(y_train), 1))) * 100

    logger.info(f"Model Accuracy: {acc:.3f}")
    logger.info(f"MAE: {mae:.3f}, MSE: {mse:.3f}, MAPE: {mape:.2f}%")

    # Load today's features
    logger.info(f"Loading today’s features from {today_path}")
    today_df = pd.read_csv(today_path)
    X_today = today_df[numeric_cols]

    # Make predictions for today
    today_prob = model.predict_proba(X_today)[:, 1]

    result_df = pd.DataFrame({
        "Game Date": today_df["game_date"],
        "Home Team": today_df["home_team"],
        "Away Team": today_df["away_team"],
        "Win Probability": today_prob,
        "Prediction": np.where(today_prob > 0.5,
                               "Pick: " + today_df["home_team"],
                               "Pick: " + today_df["away_team"])
    })

    # Save readable predictions
    output_name = f"readable_win_predictions_for_{today_df['game_date'].iloc[0]}_using_{datetime.today().strftime('%Y-%m-%d')}.csv"
    output_path = PROCESSED_DIR / output_name
    # output_path = os.path.join("C:/Users/roman/baseball_forecast_project/data/processed", output_name)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    return result_df


if __name__ == "__main__":
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    historical_path = PROCESSED_DIR / f"historical_main_features.csv"
    today_path = PROCESSED_DIR / f"main_features_{today_str}.csv"

    try:
        train_model(historical_path, today_path)
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
    

    # cd C:\Users\roman\baseball_forecast_project\modeling
    # python train_model.py
