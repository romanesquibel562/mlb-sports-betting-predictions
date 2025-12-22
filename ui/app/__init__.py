# ui/app/__init__.py

from dotenv import load_dotenv
import os
from pathlib import Path
from flask import Flask


def create_app():
    load_dotenv()  # Load environment variables from .env file
    app = Flask(__name__)

    # __file__ = ui/app/__init__.py
    # parents[1] = ui/
    # This is the UI-specific data folder (e.g. team_power_rankings_*.csv)
    ui_data_dir = (Path(__file__).resolve().parents[1] / "data").resolve()
    app.config["DATA_DIR"] = str(ui_data_dir)

    # Project root: .../baseball_forecast_project
    project_root = Path(__file__).resolve().parents[2]

    # Processed MLB pipeline output directory (main_features_*.csv, etc.)
    processed_dir = (project_root / "data" / "processed").resolve()
    app.config["PROCESSED_DATA_DIR"] = str(processed_dir)

    # NEW: predictions directory for readable_win_predictions_for_*.csv
    predictions_dir = (project_root / "data" / "predictions").resolve()
    app.config["PREDICTIONS_DATA_DIR"] = str(predictions_dir)

    # Gemini API key from .env
    app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

    from .routes_ui import ui_bp
    from .routes_api import api_bp

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app

