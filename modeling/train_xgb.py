# train_xgb.py

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def train_model(historical_path, today_path):
    logger.info(f"Loading historical dataset from {historical_path}")
    historical_df = pd.read_csv(historical_path)

    # Filter numeric features
    non_feature_cols = [
        "actual_winner", "game_date", "home_team", "away_team",
        "home_pitcher", "away_pitcher", "home_pitcher_full_name", "away_pitcher_full_name"
    ]
    numeric_cols = [col for col in historical_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(historical_df[col])]
    X_train = historical_df[numeric_cols]
    y_train = (historical_df["actual_winner"] == historical_df["home_team"]).astype(int)

    logger.info(f"Training on {len(X_train)} historical games with {X_train.shape[1]} features")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # XGBoost with calibration
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    base_model.fit(X_train, y_train)

   # === Plot feature importances (Top 10) ===
    importances = base_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importance.png')
    plt.close()

    # === Calibrate the model ===
    model = CalibratedClassifierCV(estimator=base_model, cv=5)
    model.fit(X_train, y_train)

    # === Internal validation ===
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / np.maximum(np.abs(y_val), 1))) * 100

    logger.info(f"Model Accuracy: {acc:.3f}")
    logger.info(f"MAE: {mae:.3f}, MSE: {mse:.3f}, MAPE: {mape:.2f}%")

    # === Save metrics to log file ===
    metrics_path = PROCESSED_DIR / "model_metrics_log.csv"
    metrics_df = pd.DataFrame([{
        "date": datetime.today().strftime('%Y-%m-%d'),
        "accuracy": acc,
        "mae": mae,
        "mse": mse,
        "mape": mape
    }])
    if metrics_path.exists():
        metrics_df.to_csv(metrics_path, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_path, index=False)

    # === Calibration curve ===
    prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title('Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'calibration_curve.png')
    plt.close()

    # === Histogram of predicted probabilities ===
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=10, edgecolor='black')
    plt.title('Histogram of Calibrated Probabilities')
    plt.xlabel('Predicted Home Win Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'probability_histogram.png')
    plt.close()

    # === Load today’s features and make predictions ===
    logger.info(f"Loading today’s features from {today_path}")
    today_df = pd.read_csv(today_path)
    X_today = today_df.reindex(columns=X_train.columns, fill_value=0)

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

    matchup_date = today_df['game_date'].iloc[0]
    statcast_date = datetime.today().strftime('%Y-%m-%d')
    output_name = f"readable_win_predictions_for_{matchup_date}_using_{statcast_date}.csv"
    output_path = PROCESSED_DIR / output_name
    result_df.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to: {output_path}")
    return result_df

# Run standalone
if __name__ == "__main__":
    today_str = datetime.today().strftime('%Y-%m-%d')
    historical_path = PROCESSED_DIR / "historical_main_features.csv"
    today_path = PROCESSED_DIR / f"main_features_{today_str}.csv"

    try:
        train_model(historical_path, today_path)
    except Exception as e:
        logger.error(f"Error in train_model: {e}")

# cd C:\Users\roman\baseball_forecast_project\modeling
# python train_xgb.py
