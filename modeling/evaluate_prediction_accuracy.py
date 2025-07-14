# evaluate_prediction_accuracy.py

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

# Map abbreviations to full team names for comparison
team_name_map = {
    'BOS': 'Red Sox', 'NYY': 'Yankees', 'TOR': 'Blue Jays', 'BAL': 'Orioles', 'TB': 'Rays',
    'CLE': 'Guardians', 'CWS': 'White Sox', 'KC': 'Royals', 'DET': 'Tigers', 'MIN': 'Twins',
    'HOU': 'Astros', 'SEA': 'Mariners', 'TEX': 'Rangers', 'LAA': 'Angels', 'OAK': 'Athletics',
    'ATL': 'Braves', 'MIA': 'Marlins', 'NYM': 'Mets', 'PHI': 'Phillies', 'WSH': 'Nationals',
    'MIL': 'Brewers', 'STL': 'Cardinals', 'CHC': 'Cubs', 'PIT': 'Pirates', 'CIN': 'Reds',
    'LAD': 'Dodgers', 'SF': 'Giants', 'SD': 'Padres', 'COL': 'Rockies', 'ARI': 'D-backs'
}

def evaluate_predictions(pred_date: str):
    pred_file = DATA_DIR / f"readable_win_predictions_for_{pred_date}_using_{pred_date}.csv"
    actual_file = DATA_DIR / f"historical_results_{pred_date}.csv"

    if not pred_file.exists() or not actual_file.exists():
        print(f"Missing prediction or actuals file for {pred_date}")
        return None

    pred_df = pd.read_csv(pred_file)
    pred_df['Predicted Winner'] = pred_df['Prediction'].str.replace('Pick: ', '').str.strip()
    pred_df['home_team'] = pred_df['Home Team'].map(team_name_map)
    pred_df['away_team'] = pred_df['Away Team'].map(team_name_map)
    pred_df['Predicted Winner'] = pred_df['Predicted Winner'].map(team_name_map)

    actual_df = pd.read_csv(actual_file)
    pred_df['game_date'] = pd.to_datetime(pred_df['Game Date']).dt.date
    actual_df['game_date'] = pd.to_datetime(actual_df['game_date']).dt.date

    merged = pd.merge(pred_df, actual_df, on=['game_date', 'home_team', 'away_team'], how='inner')
    merged['Correct'] = merged['Predicted Winner'] == merged['winner']

    correct = merged['Correct'].sum()
    total = len(merged)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0

    print(f"\nEvaluating Predictions for: {pred_date}")
    print(f"Total Games Evaluated: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Accuracy: {accuracy}%\n")

    wrong = merged[~merged['Correct']]
    if not wrong.empty:
        print("Incorrect Predictions:")
        print(wrong[['game_date', 'home_team', 'away_team', 'Predicted Winner', 'winner']].to_string(index=False))

    return {
        'date': pred_date,
        'games': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }

def evaluate_range(start_date: str, end_date: str):
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    summary = []

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        result = evaluate_predictions(date_str)
        if result:
            summary.append(result)
        current += timedelta(days=1)

    if summary:
        summary_df = pd.DataFrame(summary)
        print("\nOverall Accuracy Summary:")
        print(summary_df.to_string(index=False))
        log_path = DATA_DIR / "prediction_accuracy_log.csv"
        summary_df.to_csv(log_path, index=False)
        print(f"\nAccuracy log saved to: {log_path}")
    else:
        print("No valid predictions or actuals found for the selected range.")

if __name__ == "__main__":
    # Option 1: Run single-day evaluation (yesterday by default)
    single_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    evaluate_predictions(single_date)

    # Option 2: Run evaluation across a full range to yesterday
    evaluate_range("2025-07-03", single_date)

    # cd C:\Users\roman\baseball_forecast_project\modeling
    # python evaluate_prediction_accuracy.py