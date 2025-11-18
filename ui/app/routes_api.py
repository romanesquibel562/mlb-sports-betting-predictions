# ui/app/routes_api.py

from .data_loader import load_rankings
from flask import Blueprint, current_app, jsonify, request

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

# test example:

# cd C:\Users\roman\baseball_forecast_project\ui
# flask --app wsgi run --debug
# http://127.0.0.1:5000/api/test_gemini_summary