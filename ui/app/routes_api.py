# ui/app/routes_api.py

from .data_loader import load_rankings
from flask import Blueprint, current_app, jsonify, request
import pandas as pd

from .data_loader import load_rankings
from .gemini_summarizer import summarize_game

api_bp = Blueprint("api", __name__)

@api_bp.get("/power-rankings")
def api_power_rankings():
    date_str = request.args.get("date")
    data = load_rankings(current_app.config["DATA_DIR"], date_str)
    return jsonify(data)


@api_bp.route("/test_gemini_key")
def test_gemini_key():
    return jsonify({
        "has_key": bool(current_app.config.get("GEMINI_API_KEY"))
    })

@api_bp.get("/test_gemini_summary")
def test_gemini_summary():
    import pandas as pd
    from pathlib import Path
    from .gemini_summarizer import summarize_game

    data_dir = Path(current_app.config["PROCESSED_DATA_DIR"])

    csv_path = list(data_dir.glob("main_features_*.csv"))
    if not csv_path:
        return jsonify({"error": "No main_features_*.csv file found"}), 404
    
    csv_path = csv_path[0]

    df = pd.read_csv(csv_path)

    game_row = df.iloc[0].to_dict()

    summary = summarize_game(game_row)

    return jsonify({
        "csv_file": str(csv_path),
        "summary": summary
    })

@api_bp.get("/game_summary/<int:row_idx>")
def api_game_summary(row_idx: int):
    """
    Return a Gemini summary for a specific game identified by its row index
    in the latest main_features_*.csv file.
    """
    import pandas as pd
    from pathlib import Path
    from .data_loader import load_rankings
    from .gemini_summarizer import summarize_game

    data_dir = Path(current_app.config["PROCESSED_DATA_DIR"])

    csv_list = sorted(data_dir.glob("main_features_*.csv"))
    if not csv_list:
        return jsonify({"error": f"No main_features_*.csv file found in {data_dir}"}), 404
    
    csv_path = csv_list[-1]

    df = pd.read_csv(csv_path)

    if row_idx < 0 or row_idx >= len(df):
        return jsonify({"error": f"Invalid row index {row_idx}"}), 400

    game_row = df.iloc[row_idx].to_dict()
    summary = summarize_game(game_row) 

    return jsonify({
        "csv_file": str(csv_path),
        "row_index": row_idx,
        "game_date": game_row.get("game_date"),
        "home_team": game_row.get("home_team"),
        "away_team": game_row.get("away_team"),
        "summary": summary,
    })


# test example:

# cd C:\Users\roman\baseball_forecast_project\ui
# flask --app wsgi run --debug
# http://127.0.0.1:5000/api/test_gemini_summary