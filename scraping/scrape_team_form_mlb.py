# scraping/scrape_team_form_mlb.py

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import re  # --- FIX: cleaner

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEAM_FORM_FMT = "team_form_{d}.csv"  # per-day output
RECENT_SYMLINK = PROCESSED_DIR / "team_recent_form.csv"

# === Team normalization ===
TEAM_NAME_MAP: Dict[str, str] = {
    'New York Yankees': 'NYY','Boston Red Sox': 'BOS','Tampa Bay Rays': 'TB',
    'Toronto Blue Jays': 'TOR','Baltimore Orioles': 'BAL','Cleveland Guardians': 'CLE',
    'Detroit Tigers': 'DET','Kansas City Royals': 'KC','Chicago White Sox': 'CWS',
    'Minnesota Twins': 'MIN','Houston Astros': 'HOU','Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX','Los Angeles Angels': 'LAA','Oakland Athletics': 'OAK',
    'Atlanta Braves': 'ATL','Miami Marlins': 'MIA','New York Mets': 'NYM',
    'Philadelphia Phillies': 'PHI','Washington Nationals': 'WSH','Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN','Milwaukee Brewers': 'MIL','Pittsburgh Pirates': 'PIT',
    'St. Louis Cardinals': 'STL','Arizona Diamondbacks': 'ARI','Colorado Rockies': 'COL',
    'Los Angeles Dodgers': 'LAD','San Diego Padres': 'SD','San Francisco Giants': 'SF'
}

def clean_team_key(s: str) -> str:
    up = str(s).upper().strip()
    up = re.sub(r"[.'’]", "", up)
    up = re.sub(r"\s+", " ", up)
    up = re.sub(r"\bST\s+LOUIS\b", "ST LOUIS", up)
    up = re.sub(r"\bD ?BACKS\b", "DIAMONDBACKS", up)
    up = re.sub(r"\bLA ANGELS\b", "LOS ANGELES ANGELS", up)
    return up

# --- FIX: requests helper with retries/timeouts
def get_json(url: str, params: Optional[Dict] = None, retries: int = 3, timeout: int = 20) -> Dict:
    last_ex = None
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            last_ex = ex
            logging.warning("GET %s (attempt %d/%d) failed: %s", url, i, retries, ex)
            time.sleep(0.5 * i)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts: {last_ex}")
# --- END FIX

# --- FIX: discover first/last regular-season dates (gameType=R)
def discover_regular_season_bounds(season: int) -> tuple[date, date]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "startDate": f"{season}-01-01",
        "endDate": f"{season}-12-31",
    }
    data = get_json(url, params)
    dates = data.get("dates", [])
    if not dates:
        raise RuntimeError(f"No regular-season schedule dates found for {season}.")
    first_date = datetime.strptime(dates[0]["date"], "%Y-%m-%d").date()
    last_date = datetime.strptime(dates[-1]["date"], "%Y-%m-%d").date()
    return first_date, last_date
# --- END FIX

def fetch_team_form_for_date(target_date: date, season: int) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {
        "leagueId": "103,104",          # AL + NL
        "season": season,
        "standingsTypes": "regularSeason",
        "date": target_date.strftime("%Y-%m-%d"),
    }
    data = get_json(url, params)
    teams: List[Dict] = []
    for record in data.get("records", []):
        for trec in record.get("teamRecords", []):
            raw_name = trec.get("team", {}).get("name", "")
            mapped = TEAM_NAME_MAP.get(raw_name)
            if mapped is None:  # try a couple trivial normalizations
                mapped = TEAM_NAME_MAP.get(raw_name.replace("St.", "St").replace("St ", "St "))
            code = mapped if mapped else clean_team_key(raw_name)
            teams.append({
                "team": code,
                "wins": trec.get("wins"),
                "losses": trec.get("losses"),
                "run_diff": trec.get("runDifferential"),
                "streak": (trec.get("streak") or {}).get("streakCode", ""),
                "games_played": trec.get("gamesPlayed"),
                "win_pct": trec.get("winningPercentage"),
            })

    df = pd.DataFrame(teams)

    # --- FIX: if no standings (off day / pre-season), return empty safely
    if df.empty:
        return df
    # --- END FIX

    # Coerce numerics & tidy strings
    for c in ("wins", "losses", "run_diff", "games_played"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "win_pct" in df.columns:
        df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce")
    if "streak" in df.columns:
        df["streak"] = df["streak"].astype(str).str.strip()

    df["team"] = df["team"].astype(str).str.upper().str.strip()
    return df

def write_team_form(df: pd.DataFrame, d: date) -> Optional[Path]:
    if df.empty:
        logging.info("No standings for %s; skipping file.", d.isoformat())
        return None
    out_path = PROCESSED_DIR / TEAM_FORM_FMT.format(d=d.strftime("%Y-%m-%d"))
    df.to_csv(out_path, index=False, encoding="utf-8")
    # also keep a convenience "recent" copy
    df.to_csv(RECENT_SYMLINK, index=False, encoding="utf-8")
    return out_path

# --- FIX: main driver with smart defaults
def run(start: Optional[str], end: Optional[str], season: Optional[int], fill_to_end: bool) -> None:
    if season is None:
        season = date.today().year

    # discover season bounds (Opening Day, last day)
    first_day, last_day = discover_regular_season_bounds(season)

    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    else:
        start_dt = first_day  # default to Opening Day

    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    else:
        end_dt = last_day if fill_to_end else start_dt

    # keep in-bounds
    if start_dt < first_day:
        logging.info("Adjusting start to Opening Day: %s", first_day)
        start_dt = first_day
    if end_dt > last_day:
        logging.info("Adjusting end to last regular-season date: %s", last_day)
        end_dt = last_day

    if end_dt < start_dt:
        raise ValueError("end date must be on/after start date")

    logging.info("Scraping team form from %s to %s (season=%d)", start_dt, end_dt, season)

    current, total = start_dt, 0
    while current <= end_dt:
        try:
            df = fetch_team_form_for_date(current, season)
            outp = write_team_form(df, current)
            if outp:
                total += 1
                logging.info("Saved %d rows to %s", len(df), outp.name)
            else:
                logging.info("No file for %s (no standings).", current)
        except Exception as ex:
            logging.error("Failed on %s: %s", current.isoformat(), ex)
        time.sleep(0.25)  # be polite
        current += timedelta(days=1)

    logging.info("Done. Generated %d file(s) into %s", total, PROCESSED_DIR)
# --- END FIX

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-day MLB team form CSVs (team_form_YYYY-MM-DD.csv). "
                    "Defaults to Opening Day if --start omitted. Use --fill-to-end to go through the last regular-season date."
    )
    p.add_argument("--start", type=str, help="Start date (YYYY-MM-DD). Defaults to Opening Day.")
    p.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Defaults to --start, or last day with --fill-to-end.")
    p.add_argument("--season", type=int, help="Season year, e.g. 2025. Defaults to current year.")
    p.add_argument("--fill-to-end", action="store_true", help="If set and --end omitted, fill through last regular-season date.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.start, args.end, args.season, args.fill_to_end)
    
# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrape_team_form_mlb.py
