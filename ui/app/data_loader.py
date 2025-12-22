# ui/app/data_loader.py

from pathlib import Path
from functools import lru_cache
import re
import pandas as pd
from datetime import datetime

# Match team_power_rankings_YYYY-MM-DD.csv
PATTERN = re.compile(r"team_power_rankings_(\d{4}-\d{2}-\d{2})\.csv$", re.I)

def list_ranking_files(data_dir: Path):
    """Return list of (date_str, Path) newest → oldest."""
    matches = []
    for f in data_dir.glob("team_power_rankings_*.csv"):
        m = PATTERN.search(f.name)
        if m:
            matches.append((m.group(1), f))
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches

@lru_cache(maxsize=32)
def load_rankings(data_dir_str: str, date_str: str | None = None):
    """
    Load rankings for a given date (YYYY-MM-DD) or latest if date_str is None.
    Returns dict for UI & API.
    """
    data_dir = Path(data_dir_str)
    files = list_ranking_files(data_dir)

    if not files:
        raise FileNotFoundError(f"No team_power_rankings_*.csv files found in {data_dir}")

    if date_str:
        file_map = {d: p for d, p in files}
        if date_str not in file_map:
            raise FileNotFoundError(f"No rankings file found for date {date_str}")
        chosen_date, chosen_path = date_str, file_map[date_str]
    else:
        chosen_date, chosen_path = files[0]

    df = pd.read_csv(chosen_path)

    # If you know your columns, you can enforce types here
    # Example: rank and power_score as numeric
    for numeric_col in ("rank", "power_score", "gpa"):
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    records = df.to_dict(orient="records")

    return {
        "date": chosen_date,
        "records": records,
        "columns": list(df.columns),
        "available_dates": [d for d, _ in files],
    }

def _norm_team(s: str) -> str:
    """Normalize team string for matching."""
    return s.strip().upper()

@lru_cache(maxsize=1)
def load_historical_results(processed_dir_str: str) -> pd.DataFrame:
    """
    Load data/processed/historical_results.csv with columns:
    game_date, home_team, away_team, winner
    """
    processed_dir = Path(processed_dir_str)
    path = processed_dir / "historical_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}")
    df = pd.read_csv(path, parse_dates=["game_date"])
    for c in ["home_team", "away_team", "winner"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_norm_team)
    df["matchup_label"] = df["away_team"] + " @ " + df["home_team"]
    return df

def list_results_for_lookup(processed_dir_str: str) -> list[dict]:
    """
    Return rows (newest→oldest) for the lookup table:
    [{game_date, home_team, away_team, matchup_label}, ...]
    """
    df = load_historical_results(processed_dir_str).copy()
    df = df.sort_values("game_date", ascending=False)
    return df[["game_date", "home_team", "away_team", "matchup_label"]].to_dict("records")

def get_prior_home_outcomes(
    processed_dir_str: str,
    game_date: str | datetime,
    home_team: str,
    away_team: str,
    last_n: int = 10,
) -> dict:
    """
    For a given matchup (away @ home) on game_date, compute outcomes of all
    *prior* games with the same orientation (same home, same away).
    Returns counts of HOME WINS vs HOME LOSSES and the last N rows.
    """
    df = load_historical_results(processed_dir_str)
    as_of = pd.to_datetime(game_date)
    H, A = _norm_team(home_team), _norm_team(away_team)

    prior = df[
        (df["home_team"] == H) &
        (df["away_team"] == A) &
        (df["game_date"] < as_of)
    ].copy()

    if prior.empty:
        return {
            "has_data": False,
            "home_team": H,
            "away_team": A,
            "as_of": as_of.date().isoformat(),
        }

    prior = prior.sort_values("game_date", ascending=False)
    total = int(len(prior))
    home_wins = int((prior["winner"] == H).sum())
    home_losses = total - home_wins

    keep_cols = [c for c in ["game_date", "away_team", "home_team", "winner"] if c in prior.columns]
    last_games = prior.head(last_n)[keep_cols].to_dict("records")

    return {
        "has_data": True,
        "home_team": H,
        "away_team": A,
        "as_of": as_of.date().isoformat(),
        "total_games": total,
        "home_wins": home_wins,
        "home_losses": home_losses,
        "last_games": last_games,
    }  