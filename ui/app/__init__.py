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
    data_dir = (Path(__file__).resolve().parents[1] / "data").resolve()
    app.config["DATA_DIR"] = str(data_dir)

    # processed MLB pipeline output directory
    processed_dir = (Path(__file__).resolve().parents[2] / "data" / "processed").resolve()
    app.config["PROCESSED_DATA_DIR"] = str(processed_dir)

    app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    from .routes_ui import ui_bp
    from .routes_api import api_bp

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app

