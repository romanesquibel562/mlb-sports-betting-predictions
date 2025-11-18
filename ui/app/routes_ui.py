# ui/app/routes_ui.py

from flask import Blueprint, render_template, request, current_app
from .data_loader import load_rankings

ui_bp = Blueprint("ui", __name__)

@ui_bp.get("/")
def home():
    # Redirect conceptually to power rankings; for now we just render that page
    return power_rankings()

@ui_bp.get("/power-rankings")
def power_rankings():
    # Optional ?date=YYYY-MM-DD
    date_str = request.args.get("date")
    data = load_rankings(current_app.config["DATA_DIR"], date_str)

    return render_template(
        "power_rankings.html",
        page_title="MLB Power Rankings",
        date=data["date"],
        table_records=data["records"],
        columns=data["columns"],
        available_dates=data["available_dates"],
    )