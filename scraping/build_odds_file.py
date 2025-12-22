# scraping/build_odds_file.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import math
import pandas as pd

# ----------------------------- config & paths ------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw" / "odds"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- team normalization -------------------------
TEAM_CANON = {
    # long names
    "los angeles dodgers": "Los Angeles Dodgers",
    "san diego padres": "San Diego Padres",
    "san francisco giants": "San Francisco Giants",
    "arizona diamondbacks": "Arizona Diamondbacks",
    "colorado rockies": "Colorado Rockies",
    "new york yankees": "New York Yankees",
    "new york mets": "New York Mets",
    "boston red sox": "Boston Red Sox",
    "philadelphia phillies": "Philadelphia Phillies",
    "chicago cubs": "Chicago Cubs",
    "atlanta braves": "Atlanta Braves",
    "miami marlins": "Miami Marlins",
    "washington nationals": "Washington Nationals",
    "st louis cardinals": "St. Louis Cardinals",
    "milwaukee brewers": "Milwaukee Brewers",
    "cincinnati reds": "Cincinnati Reds",
    "pittsburgh pirates": "Pittsburgh Pirates",
    "houston astros": "Houston Astros",
    "texas rangers": "Texas Rangers",
    "seattle mariners": "Seattle Mariners",
    "oakland athletics": "Oakland Athletics",
    "los angeles angels": "Los Angeles Angels",
    "detroit tigers": "Detroit Tigers",
    "cleveland guardians": "Cleveland Guardians",
    "kansas city royals": "Kansas City Royals",
    "minnesota twins": "Minnesota Twins",
    "toronto blue jays": "Toronto Blue Jays",
    "tampa bay rays": "Tampa Bay Rays",
    "baltimore orioles": "Baltimore Orioles",
    "chicago white sox": "Chicago White Sox",

    # abbreviations → long
    "lad": "Los Angeles Dodgers", "sd": "San Diego Padres", "sfg": "San Francisco Giants",
    "ari": "Arizona Diamondbacks", "col": "Colorado Rockies", "nyy": "New York Yankees",
    "nym": "New York Mets", "bos": "Boston Red Sox", "phi": "Philadelphia Phillies",
    "chc": "Chicago Cubs", "atl": "Atlanta Braves", "mia": "Miami Marlins",
    "was": "Washington Nationals", "wsh": "Washington Nationals", "stl": "St. Louis Cardinals",
    "mil": "Milwaukee Brewers", "cin": "Cincinnati Reds", "pit": "Pittsburgh Pirates",
    "hou": "Houston Astros", "tex": "Texas Rangers", "sea": "Seattle Mariners",
    "oak": "Oakland Athletics", "laa": "Los Angeles Angels", "det": "Detroit Tigers",
    "cle": "Cleveland Guardians", "kc": "Kansas City Royals", "kcr": "Kansas City Royals",
    "min": "Minnesota Twins", "tor": "Toronto Blue Jays", "tb": "Tampa Bay Rays",
    "tbr": "Tampa Bay Rays", "bal": "Baltimore Orioles", "cws": "Chicago White Sox",

    # nicknames (no city) → long
    "yankees": "New York Yankees",
    "mets": "New York Mets",
    "red sox": "Boston Red Sox",
    "phillies": "Philadelphia Phillies",
    "cubs": "Chicago Cubs",
    "braves": "Atlanta Braves",
    "marlins": "Miami Marlins",
    "nationals": "Washington Nationals",
    "cardinals": "St. Louis Cardinals",
    "brewers": "Milwaukee Brewers",
    "reds": "Cincinnati Reds",
    "pirates": "Pittsburgh Pirates",
    "astros": "Houston Astros",
    "rangers": "Texas Rangers",
    "mariners": "Seattle Mariners",
    "athletics": "Oakland Athletics",
    "angels": "Los Angeles Angels",
    "tigers": "Detroit Tigers",
    "guardians": "Cleveland Guardians",
    "royals": "Kansas City Royals",
    "twins": "Minnesota Twins",
    "blue jays": "Toronto Blue Jays",
    "rays": "Tampa Bay Rays",
    "orioles": "Baltimore Orioles",
    "white sox": "Chicago White Sox",
    "giants": "San Francisco Giants",
    "padres": "San Diego Padres",
    "rockies": "Colorado Rockies",
    "dodgers": "Los Angeles Dodgers",
    "diamondbacks": "Arizona Diamondbacks",
    "d backs": "Arizona Diamondbacks",
    "d-backs": "Arizona Diamondbacks",
}

def normalize_team(s: str) -> str:
    if not isinstance(s, str):
        return s
    key = re.sub(r"[^a-z ]+", " ", s.lower()).strip()
    return TEAM_CANON.get(key, s.title())

# ----------------------------- odds helpers -------------------------------
def implied_from_american(ml: int | float | None) -> float | None:
    if ml is None or (isinstance(ml, float) and math.isnan(ml)):
        return None
    ml = int(ml)
    return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)

def american_from_prob(p: float | None) -> int | None:
    """Convert probability (0-1) to American moneyline (nearest 5)."""
    if p is None or not (0 < p < 1):
        return None
    # fair decimal odds
    dec = 1.0 / p
    # american
    if dec >= 2.0:
        ml = int(round((dec - 1.0) * 100 / 5.0) * 5)  # positive ML
    else:
        ml = int(round(-100 / (dec - 1.0) / 5.0) * 5)  # negative ML
    return ml

def add_vig(p_home: float, p_away: float, vig=0.045):
    """
    Apply a simple proportional vig so p_home' + p_away' = 1 + vig.
    Then renormalize to 1. (Equivalent to adding margin to the book.)
    """
    # Start from fair probs (sum=1); inflate both by same factor
    inflated_home = p_home * (1 + vig)
    inflated_away = p_away * (1 + vig)
    s = inflated_home + inflated_away
    # renormalize to 1 (keeps relative inflation but presents realistic probs)
    adj_home = inflated_home / s
    adj_away = inflated_away / s
    return adj_home, adj_away

# ----------------------------- builders -----------------------------------
def load_main_features(date_str: str) -> pd.DataFrame:
    mf_path = PROCESSED_DIR / f"main_features_{date_str}.csv"
    if not mf_path.exists():
        raise FileNotFoundError(f"Missing {mf_path}")
    df = pd.read_csv(mf_path)
    need = {"away_team", "home_team"}
    if not need.issubset(df.columns):
        raise ValueError("main_features is missing away_team/home_team")
    df["away_team"] = df["away_team"].astype(str).apply(normalize_team)
    df["home_team"] = df["home_team"].astype(str).apply(normalize_team)
    if "game_date" not in df.columns:
        df["game_date"] = date_str
    return df[["game_date", "away_team", "home_team"]].drop_duplicates()

def try_read_model_preds(date_str: str) -> pd.DataFrame | None:
    # readable_win_predictions_for_{date}_using_*.csv
    files = sorted(PROCESSED_DIR.glob(f"readable_win_predictions_for_{date_str}_using_*.csv"))
    if not files:
        return None
    dp = pd.read_csv(files[0])
    need = {"Game Date", "Home Team", "Away Team", "Win Probability"}
    if not need.issubset(dp.columns):
        return None
    dp = dp.rename(columns={
        "Game Date": "game_date",
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Win Probability": "home_win_prob"
    })
    dp["home_team"] = dp["home_team"].astype(str).apply(normalize_team)
    dp["away_team"] = dp["away_team"].astype(str).apply(normalize_team)
    return dp[["game_date", "home_team", "away_team", "home_win_prob"]]

def try_read_manual_raw(date_str: str) -> pd.DataFrame | None:
    # Allow user to drop odds at data/raw/odds/odds_{date}.csv (already normalized or close)
    raw_path = RAW_DIR / f"odds_{date_str}.csv"
    if not raw_path.exists():
        return None
    do = pd.read_csv(raw_path)
    # minimal expected columns
    cols = {"game_date", "away_team", "home_team", "sportsbook", "away_ml", "home_ml", "fetched_at"}
    missing = cols - set(do.columns)
    if missing:
        raise ValueError(f"Manual odds file missing columns: {missing}")
    for c in ("home_team", "away_team"):
        do[c] = do[c].astype(str).apply(normalize_team)
    return do[list(cols)]

def build_dummy_odds(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: produce reasonable lines around a coin flip:
      home ~ -115 / away ~ +105 (random small jitter)
    """
    import random
    rows = []
    for r in df_games.to_dict(orient="records"):
        # tiny tilt so they're not all identical
        tilt = random.uniform(-0.03, 0.03)
        p_home = min(max(0.5 + tilt, 0.40), 0.60)
        p_away = 1 - p_home
        p_home_vig, p_away_vig = add_vig(p_home, p_away, vig=0.045)

        home_ml = american_from_prob(p_home_vig)
        away_ml = american_from_prob(p_away_vig)

        rows.append({
            "game_date": r["game_date"],
            "away_team": r["away_team"],
            "home_team": r["home_team"],
            "sportsbook": "DemoBook",
            "away_ml": away_ml,
            "home_ml": home_ml,
            "fetched_at": None,
        })
    return pd.DataFrame(rows)

def build_model_based_odds(df_games: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    df = df_games.merge(preds, on=["game_date", "home_team", "away_team"], how="left")
    rows = []
    for r in df.to_dict(orient="records"):
        ph = r.get("home_win_prob")
        if pd.isna(ph):
            rows.append({
                "game_date": r["game_date"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "sportsbook": "ModelLine",
                "away_ml": None,
                "home_ml": None,
                "fetched_at": None,
            })
            continue
        try:
            ph = float(ph)
        except Exception:
            ph = None
        if ph is None or not (0 < ph < 1):
            ah = None
            aa = None
        else:
            pa = 1 - ph
            ph_v, pa_v = add_vig(ph, pa, vig=0.045)
            ah = american_from_prob(ph_v)
            aa = american_from_prob(pa_v)

        rows.append({
            "game_date": r["game_date"],
            "away_team": r["away_team"],
            "home_team": r["home_team"],
            "sportsbook": "ModelLine",
            "away_ml": aa,
            "home_ml": ah,
            "fetched_at": None,
        })
    return pd.DataFrame(rows)

# ----------------------------- main ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build odds_{date}.csv for Today+EV page.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    date_str = args.date

    games = load_main_features(date_str)

    # 1) manual raw odds (if provided)
    manual = try_read_manual_raw(date_str)
    if manual is not None:
        out = manual
    else:
        # 2) model-based odds (preferred if preds are present)
        preds = try_read_model_preds(date_str)
        if preds is not None:
            out = build_model_based_odds(games, preds)
        else:
            # 3) dummy fallback odds
            out = build_dummy_odds(games)

    out_path = PROCESSED_DIR / f"odds_{date_str}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()


# examples of running
# cd C:\Users\roman\baseball_forecast_project\scraping
# python build_odds_file.py --date 2025-10-27
