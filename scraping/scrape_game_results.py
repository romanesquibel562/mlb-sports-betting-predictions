# scrape_game_results.py

from __future__ import annotations
import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

import time
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("scrape_results")

# ------------------------ Paths ------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_MATCHUPS_DIR = BASE_DIR / "data" / "raw" / "historical_matchups"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------ Team normalization ------------------------
# Short names used across your pipeline
ALIASES = {
    # exact (Baseball-Reference link text -> short)
    "Arizona Diamondbacks": "D-backs", "Diamondbacks": "D-backs", "ARI": "D-backs", "Dbacks": "D-backs",
    "Atlanta Braves": "Braves", "ATL": "Braves",
    "Baltimore Orioles": "Orioles", "BAL": "Orioles",
    "Boston Red Sox": "Red Sox", "BOS": "Red Sox",
    "Chicago White Sox": "White Sox", "CWS": "White Sox", "Chi White Sox": "White Sox",
    "Chicago Cubs": "Cubs", "CHC": "Cubs",
    "Cincinnati Reds": "Reds", "CIN": "Reds",
    "Cleveland Guardians": "Guardians", "CLE": "Guardians",
    "Colorado Rockies": "Rockies", "COL": "Rockies",
    "Detroit Tigers": "Tigers", "DET": "Tigers",
    "Houston Astros": "Astros", "HOU": "Astros",
    "Kansas City Royals": "Royals", "KC": "Royals",
    "Los Angeles Angels": "Angels", "LAA": "Angels", "LA Angels": "Angels", "Los Angeles Angels of Anaheim": "Angels",
    "Los Angeles Dodgers": "Dodgers", "LAD": "Dodgers", "LA Dodgers": "Dodgers",
    "Miami Marlins": "Marlins", "MIA": "Marlins", "Florida Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers", "MIL": "Brewers",
    "Minnesota Twins": "Twins", "MIN": "Twins",
    "New York Mets": "Mets", "NYM": "Mets",
    "New York Yankees": "Yankees", "NYY": "Yankees",
    "Oakland Athletics": "Athletics", "OAK": "Athletics", "Athletics": "Athletics", "A's": "Athletics",
    "Philadelphia Phillies": "Phillies", "PHI": "Phillies",
    "Pittsburgh Pirates": "Pirates", "PIT": "Pirates",
    "San Diego Padres": "Padres", "SD": "Padres", "SDP": "Padres",
    "San Francisco Giants": "Giants", "SF": "Giants", "SFG": "Giants",
    "Seattle Mariners": "Mariners", "SEA": "Mariners",
    "St. Louis Cardinals": "Cardinals", "STL": "Cardinals",
    "Tampa Bay Rays": "Rays", "TB": "Rays", "Tampa Bay Devil Rays": "Rays",
    "Texas Rangers": "Rangers", "TEX": "Rangers",
    "Toronto Blue Jays": "Blue Jays", "TOR": "Blue Jays",
    "Washington Nationals": "Nationals", "WSH": "Nationals", "WAS": "Nationals", "Montreal Expos": "Nationals",
}

def norm_team(s: str) -> str | None:
    if s is None:
        return None
    ss = str(s).strip()
    return ALIASES.get(ss, ALIASES.get(ss.title(), ss)).strip() if ss else None


def _soup_via_selenium(url: str) -> BeautifulSoup:
    """Fallback fetch using headless Chrome (mimics a real browser)."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(1.5)  # let the page fully render
        html = driver.page_source
        return BeautifulSoup(html, "html.parser")
    finally:
        driver.quit()

# ------------------------ Core scraping ------------------------
def fetch_results_br(day: date) -> list[tuple[str, str]]:
    """
    Returns list of (winner, loser) short names for a given day from Baseball-Reference.
    First tries requests with realistic headers + cookies; if 403/blocked, falls back to Selenium.
    """
    url = f"https://www.baseball-reference.com/boxes/?year={day.year}&month={day.month}&day={day.day}"

    # --- Try requests first with a browser-like session and headers
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.baseball-reference.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    # preflight to set cookies
    try:
        session.get("https://www.baseball-reference.com", headers=headers, timeout=20)
        # small randomized delay to be polite
        time.sleep(0.6 + random.random() * 0.8)

        r = session.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
        else:
            # If blocked (e.g., 403), jump to Selenium fallback
            soup = _soup_via_selenium(url)
    except Exception:
        # Network error or 403 during preflight -> fallback
        soup = _soup_via_selenium(url)

    pairs: list[tuple[str, str]] = []
    for game in soup.select("div.game_summary"):
        w_tag = game.select_one("tr.winner td a")
        l_tag = game.select_one("tr.loser td a")
        if not w_tag or not l_tag:
            continue
        w = norm_team(w_tag.text)
        l = norm_team(l_tag.text)
        if w and l:
            pairs.append((w, l))
    return pairs

def scrape_results_for_date(day: date, matchup_csv_path: Path, out_dir: Path) -> Path | None:
    """
    Reads the day's historical_matchups_YYYY-MM-DD.csv and matches winners from BR.
    Writes data/processed/historical_results_YYYY-MM-DD.csv
    """
    out_path = out_dir / f"historical_results_{day}.csv"
    if out_path.exists():
        log.info(f"Skip {day}: results already exist.")
        return out_path

    if not matchup_csv_path.exists():
        log.info(f"No matchup file for {day}: {matchup_csv_path}")
        return None

    df_m = pd.read_csv(matchup_csv_path)
    if df_m.empty:
        log.info(f"Empty matchup file for {day}")
        return None

    # Normalize teams in matchup file
    for c in ["home_team", "away_team"]:
        df_m[c] = df_m[c].map(norm_team)

    winners = fetch_results_br(day)  # list[(winner, loser)]
    if not winners:
        log.warning(f"No parsed winners from BR for {day}")
        return None

    # Build set to speed membership lookups
    day_pairs = {(row["home_team"], row["away_team"]) for _, row in df_m.iterrows()}

    rows = []
    # For each known winner-loser pair, see if it matches either orientation in the matchup list,
    # then emit the standardized (home_team, away_team, winner).
    for w, l in winners:
        if (w, l) in day_pairs or (l, w) in day_pairs:
            # Decide orientation by checking which tuple exists in matchups
            if (w, l) in day_pairs:
                # winner was home team
                rows.append({"game_date": day, "home_team": w, "away_team": l, "winner": w})
            elif (l, w) in day_pairs:
                # winner was away team
                rows.append({"game_date": day, "home_team": l, "away_team": w, "winner": w})

    if not rows:
        log.warning(f"No results matched the matchups for {day}")
        return None

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    log.info(f"Saved {len(out_df)} results to {out_path}")
    return out_path

# ------------------------ Orchestration ------------------------
def daterange(d0: date, d1: date):
    """Yield dates from d0 to d1 inclusive."""
    step = timedelta(days=1)
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + step

def latest_existing_result_day() -> date | None:
    files = sorted(PROCESSED_DIR.glob("historical_results_*.csv"))
    if not files:
        return None
    # files like historical_results_YYYY-MM-DD.csv
    last = files[-1].stem.split("_")[-1]
    return date.fromisoformat(last)

def consolidate_results() -> Path | None:
    """Concat all historical_results_*.csv -> historical_results.csv (deduped, sorted)."""
    files = sorted(PROCESSED_DIR.glob("historical_results_*.csv"))
    if not files:
        log.warning("No per-day results to consolidate.")
        return None
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p, parse_dates=["game_date"])
            for c in ["home_team", "away_team", "winner"]:
                df[c] = df[c].map(norm_team)
            dfs.append(df)
        except Exception as e:
            log.warning(f"Skipping {p}: {e}")
    if not dfs:
        return None
    full = pd.concat(dfs, ignore_index=True).drop_duplicates()
    full = full.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    out = PROCESSED_DIR / "historical_results.csv"
    full.to_csv(out, index=False)
    log.info(f"Wrote consolidated {len(full)} rows -> {out}")
    return out

def run(from_day: date | None, to_day: date | None, rolling: int | None, since_last: bool):
    today = date.today()
    if rolling is not None:
        from_day = today - timedelta(days=rolling)
        to_day = today - timedelta(days=1)
    elif since_last:
        last = latest_existing_result_day()
        if last is None:
            # default: backfill 120 days if nothing exists
            from_day = today - timedelta(days=120)
        else:
            from_day = last + timedelta(days=1)
        to_day = today - timedelta(days=1)
    elif from_day and not to_day:
        to_day = today - timedelta(days=1)
    elif not from_day and not to_day:
        # default: last 30 days
        from_day = today - timedelta(days=30)
        to_day = today - timedelta(days=1)

    log.info(f"Scraping results from {from_day} to {to_day}")
    for d in daterange(from_day, to_day):
        matchup_file = RAW_MATCHUPS_DIR / f"historical_matchups_{d}.csv"
        scrape_results_for_date(d, matchup_file, PROCESSED_DIR)

    consolidate_results()

# ------------------------ CLI ------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Backfill MLB daily game results into historical_results.csv")
    ap.add_argument("--from", dest="from_day", type=str, help="Start date YYYY-MM-DD")
    ap.add_argument("--to", dest="to_day", type=str, help="End date YYYY-MM-DD")
    ap.add_argument("--rolling", type=int, help="Backfill last N days")
    ap.add_argument("--since-last", action="store_true", help="Continue from latest existing per-day results")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    from_day = date.fromisoformat(args.from_day) if args.from_day else None
    to_day = date.fromisoformat(args.to_day) if args.to_day else None
    run(from_day, to_day, args.rolling, args.since_last)


# cd C:\Users\roman\baseball_forecast_project\scraping

# Backfill from the last day you have through yesterday, then consolidate
# python scrape_game_results.py --since-last

# OR: choose a window explicitly (e.g., from Sep 21 to yesterday)
# python scrape_game_results.py --from 2025-09-21

# OR: last 80 days
# python scrape_game_results.py --rolling 80
