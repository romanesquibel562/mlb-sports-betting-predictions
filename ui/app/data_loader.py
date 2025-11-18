# ui/app/data_loader.py

from pathlib import Path
from functools import lru_cache
import re
import pandas as pd

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