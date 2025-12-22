# ui/app/routes_ui.py

from flask import Blueprint, render_template, request, current_app, url_for, redirect
from pathlib import Path
from datetime import datetime
import pandas as pd
import re
from .gemini_summarizer import summarize_game
from .data_loader import load_rankings
from .data_loader import list_results_for_lookup, get_prior_home_outcomes
from .gemini_summarizer import summarize_game_compact

ui_bp = Blueprint("ui", __name__)

# ----------------------------- Team naming utils -----------------------------

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
    """Normalize various team inputs (abbr, ALL-CAPS, 'Pick: TEAM') → canonical long name."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"^pick:\s*", "", s, flags=re.I).strip()
    key = re.sub(r"[^a-z ]+", " ", s.lower()).strip()
    return TEAM_CANON.get(key, s.title())

def _first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default

def _pct01(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        x = float(x)
        # If value looks like percent (40, 23.5), convert to 0–1
        if x > 1.5:
            return x / 100.0
        return x
    except Exception:
        return None

TEAM_LOGO_SLUG = lambda t: re.sub(r'[^A-Za-z]+', '-', str(t).lower()).strip('-')

# ----------------------------- helpers -----------------------------

def _latest_on_or_before(folder: Path, prefix: str, dt: datetime) -> Path | None:
    """Find a file like 'prefix_YYYY-MM-DD.csv' with date <= dt, returning the latest."""
    files = sorted(folder.glob(f"{prefix}_*.csv"))
    best = None
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not m:
            continue
        fdt = datetime.strptime(m.group(1), "%Y-%m-%d")
        if fdt <= dt:
            best = f
    return best

def _available_matchup_dates(processed_dir: Path) -> list[str]:
    """List all dates present in processed/historical_results_*.csv (root of processed)."""
    dates = []
    for f in processed_dir.glob("historical_results_*.csv"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)

# --------------------------------- Home ------------------------------------

# ---- Home (canonical) ----
@ui_bp.get("/", endpoint="home")
def home():
    """Landing page with links to key views."""
    return render_template("index.html")

# ---- Backward-compat alias for old templates ----
@ui_bp.get("/index", endpoint="index")
def index_alias():
    # If any template still calls url_for('ui.index'), this will work
    return redirect(url_for("ui.home"), code=302)

# --------------------------- Power Rankings --------------------------------

@ui_bp.get("/power-rankings")
def power_rankings():
    """Team power rankings page. Optional ?date=YYYY-MM-DD query param."""
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

# --------------------------- Games (grid) ----------------------------------

@ui_bp.route("/games")
def games_index():
    # This page is now an internal bridge only.
    # If someone hits it directly, send them to Game Lookup.
    if not request.args.get("date"):
        return redirect(url_for("ui.game_lookup"), code=302)

    processed_dir = Path(current_app.config["PROCESSED_DATA_DIR"])

    dates = _available_matchup_dates(processed_dir)
    if not dates:
        return render_template("empty.html", title="Games", message="No historical results files found yet.")

    date_str = request.args.get("date", dates[0])

    r_path = processed_dir / f"historical_results_{date_str}.csv"
    if not r_path.exists():
        # If the specific date isn’t available, route users to lookup instead of erroring
        return redirect(url_for("ui.game_lookup"), code=302)

    matchups = pd.read_csv(r_path)
    matchups["home_team"] = matchups["home_team"].apply(normalize_team)
    matchups["away_team"] = matchups["away_team"].apply(normalize_team)

    if "game_date" not in matchups.columns:
        matchups["game_date"] = date_str

    pred_candidates = list(processed_dir.glob(f"readable_win_predictions_for_{date_str}_using_*.csv"))
    preds = pd.read_csv(pred_candidates[0]) if pred_candidates else pd.DataFrame()

    if not preds.empty:
        preds = preds.rename(columns={
            "Game Date": "game_date",
            "Home Team": "home_team",
            "Away Team": "away_team",
            "Win Probability": "win_prob",
            "Prediction": "pick"
        })
        preds["home_team"] = preds["home_team"].apply(normalize_team)
        preds["away_team"] = preds["away_team"].apply(normalize_team)
        if "pick" in preds.columns:
            preds["pick"] = preds["pick"].apply(normalize_team)

        matchups = matchups.merge(
            preds[["game_date", "home_team", "away_team", "win_prob", "pick"]],
            on=["game_date", "home_team", "away_team"],
            how="left"
        )

    if "away_pitcher" not in matchups.columns:
        matchups["away_pitcher"] = None
    if "home_pitcher" not in matchups.columns:
        matchups["home_pitcher"] = None

    matchups["home_logo"] = matchups["home_team"].apply(TEAM_LOGO_SLUG)
    matchups["away_logo"] = matchups["away_team"].apply(TEAM_LOGO_SLUG)

    rows = matchups.sort_values(["home_team", "away_team"]).to_dict(orient="records")
    return render_template("games_index.html", date_str=date_str, dates=dates, rows=rows)

# --------------------------- Game Hub (detail) -----------------------------

@ui_bp.route("/game/<date_str>/<away>-at-<home>")
def game_detail(date_str, away, home):
    """
    Detailed Game Hub for one matchup.

    Uses:
      - processed/historical_results_{date}.csv (drives result; OPTIONAL now)
      - processed/readable_win_predictions_for_{date}_using_*.csv (optional)
      - processed/team_form_{<=date}.csv (nearest on/before date)
      - processed/pitcher_stat_features_{<=date}.csv (nearest on/before date)
      - processed/batter_stat_features_{<=date}.csv (nearest on/before date; team aggregates)
      - processed/main_features_{date}.csv (fed to summarizer; also header fallback)
    """
    processed_dir = Path(current_app.config["PROCESSED_DATA_DIR"])
    game_date = datetime.strptime(date_str, "%Y-%m-%d")

    # --- local helpers -----------------------------------------------------
    def _first(d: dict, keys: list[str], default=None):
        for k in keys:
            if k in d and pd.notna(d[k]):
                return d[k]
        return default

    def _pct01(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        try:
            x = float(x)
            return x / 100.0 if x > 1.5 else x
        except Exception:
            return None

    def _rename_pitcher_fields(d: dict) -> dict:
        if not d:
            return d
        out = dict(d)
        out["avg_velocity"]  = _first(d, ["avg_velocity", "avg_velo", "velo_avg", "pitch_velocity_avg"])
        out["avg_spin_rate"] = _first(d, ["avg_spin_rate", "spin_rate_avg", "avg_spin"])
        out["k_rate"]        = _pct01(_first(d, ["k_rate", "k_pct", "strikeout_rate"]))
        out["bb_rate"]       = _pct01(_first(d, ["bb_rate", "bb_pct", "walk_rate"]))
        out["whiff_pct"]     = _pct01(_first(d, ["whiff_pct", "whiff_rate"]))
        out["strike_pct"]    = _pct01(_first(d, ["strike_pct", "strike_rate"]))
        return out

    def clean_name(x: str | None) -> str | None:
        if not isinstance(x, str):
            return x
        x = x.strip()
        x = x.split(" (")[0].strip()            # drop handedness tags
        x = re.sub(r"\s+", " ", x)
        return x.title()

    def _canon(x: str | None) -> str | None:
        return clean_name(x)

    def _try_set_sp_from_df(df: pd.DataFrame, home_n: str, away_n: str) -> bool:
        if df is None or df.empty:
            return False
        df = df.copy()

        # generous team columns
        home_team_candidates = [c for c in df.columns if c.lower() in {
            "home_team", "home", "hometeam", "home_name", "homeclub", "homeclubname"
        } or ("home" in c.lower() and "team" in c.lower())]
        away_team_candidates = [c for c in df.columns if c.lower() in {
            "away_team", "away", "awayteam", "away_name", "awayclub", "awayclubname"
        } or ("away" in c.lower() and "team" in c.lower())]
        if not home_team_candidates or not away_team_candidates:
            return False
        hcol, acol = home_team_candidates[0], away_team_candidates[0]
        df["home_n"] = df[hcol].astype(str).apply(normalize_team)
        df["away_n"] = df[acol].astype(str).apply(normalize_team)

        # pitcher columns
        away_pit_candidates = [c for c in df.columns if c.lower() in {
            "away_pitcher", "away_sp", "probable_pitcher_away", "away_probable_pitcher"
        } or ("away" in c.lower() and "pitch" in c.lower())]
        home_pit_candidates = [c for c in df.columns if c.lower() in {
            "home_pitcher", "home_sp", "probable_pitcher_home", "home_probable_pitcher"
        } or ("home" in c.lower() and "pitch" in c.lower())]

        match = df[(df["home_n"] == home_n) & (df["away_n"] == away_n)]
        if match.empty:
            return False
        r = match.iloc[0]
        changed = False
        if away_pit_candidates:
            val = clean_name(r[away_pit_candidates[0]])
            if isinstance(val, str) and val:
                header["away_pitcher"] = val
                changed = True
        if home_pit_candidates:
            val = clean_name(r[home_pit_candidates[0]])
            if isinstance(val, str) and val:
                header["home_pitcher"] = val
                changed = True
        return changed
    # ----------------------------------------------------------------------

    # ------------------------ Anchor matchup row --------------------------
    results_path = processed_dir / f"historical_results_{date_str}.csv"
    df_results = None
    df_anchor = None

    away_n = normalize_team(away)
    home_n = normalize_team(home)

    if results_path.exists():
        df_results = pd.read_csv(results_path)
        df_results["away_team"] = df_results["away_team"].astype(str).apply(normalize_team)
        df_results["home_team"] = df_results["home_team"].astype(str).apply(normalize_team)
        df_anchor = df_results
    else:
        # Fallback to main_features when no results exist (future/sim dates)
        mf_fallback = processed_dir / f"main_features_{date_str}.csv"
        if mf_fallback.exists():
            df_mf_anchor = pd.read_csv(mf_fallback)
            if {"away_team", "home_team"}.issubset(df_mf_anchor.columns):
                df_mf_anchor["away_team"] = df_mf_anchor["away_team"].astype(str).apply(normalize_team)
                df_mf_anchor["home_team"] = df_mf_anchor["home_team"].astype(str).apply(normalize_team)
                df_anchor = df_mf_anchor
        if df_anchor is None:
            return render_template("empty.html", title="Game",
                                   message=f"No results or main_features found for {date_str}."), 404

    row = df_anchor[(df_anchor["away_team"] == away_n) & (df_anchor["home_team"] == home_n)]
    if row.empty:
        return render_template("empty.html", title="Game", message="Matchup not found for that date.")
    header = row.iloc[0].to_dict()

    # ------------------ Enrich with Starting Pitchers -----------------------
    raw_dir = Path(current_app.config["PROCESSED_DATA_DIR"]).parents[1] / "data" / "raw"

    # 1) RAW probables
    prob_path = raw_dir / "mlb_probable_pitchers" / f"mlb_probable_pitchers_{date_str}.csv"
    if prob_path.exists():
        try:
            df_prob = pd.read_csv(prob_path)
            _try_set_sp_from_df(df_prob, home_n, away_n)
        except Exception:
            pass

    # 2) Historical matchups (correct subfolder)
    if not header.get("away_pitcher") and not header.get("home_pitcher"):
        m2_path = processed_dir / "historical_matchups" / f"historical_matchups_{date_str}.csv"
        if m2_path.exists():
            try:
                dfm2 = pd.read_csv(m2_path)
                _try_set_sp_from_df(dfm2, home_n, away_n)
            except Exception:
                pass

    # 3) Fallback to main_features for SP names
    if not header.get("away_pitcher") and not header.get("home_pitcher"):
        mf_path_for_sps = processed_dir / f"main_features_{date_str}.csv"
        if mf_path_for_sps.exists():
            try:
                dfmf_sp = pd.read_csv(mf_path_for_sps)
                _try_set_sp_from_df(dfmf_sp, home_n, away_n)
            except Exception:
                pass

    if header.get("away_pitcher"):
        header["away_pitcher"] = clean_name(header["away_pitcher"])
    if header.get("home_pitcher"):
        header["home_pitcher"] = clean_name(header["home_pitcher"])

    # ------------------------ Prediction (optional) -------------------------
    prediction = None
    pred_candidates = list(processed_dir.glob(f"readable_win_predictions_for_{date_str}_using_*.csv"))
    if pred_candidates:
        dp = pd.read_csv(pred_candidates[0])
        need = {"Away Team", "Home Team", "Win Probability"}
        if need.issubset(set(dp.columns)):
            dp["Away Team"] = dp["Away Team"].astype(str).apply(normalize_team)
            dp["Home Team"] = dp["Home Team"].astype(str).apply(normalize_team)
            dp = dp[(dp["Away Team"] == away_n) & (dp["Home Team"] == home_n)]
            if not dp.empty:
                p = dp.iloc[0]
                prediction = {
                    "win_prob": float(p["Win Probability"]),
                    "pick": normalize_team(p.get("Prediction"))
                }

    # ---------------------------- Result -----------------------------------
    result = None
    if df_results is not None:
        dr = df_results[(df_results["away_team"] == away_n) & (df_results["home_team"] == home_n)]
        if not dr.empty:
            result = dr.iloc[0].to_dict()

    # --------------------------- Team Form ---------------------------------
    form_path = _latest_on_or_before(processed_dir, "team_form", game_date)
    form = {}
    if form_path:
        df_form = pd.read_csv(form_path)
        df_form["team"] = df_form["team"].astype(str).str.strip()
        for team in (away_n, home_n):
            frow = df_form[df_form["team"].apply(normalize_team) == team]
            if not frow.empty:
                form[team] = frow.iloc[0].to_dict()

    # ----------------------- Pitcher Features ------------------------------
    pit_path = _latest_on_or_before(processed_dir, "pitcher_stat_features", game_date)
    pitchers = {}
    if pit_path:
        dpf = pd.read_csv(pit_path)
        name_col = None
        for c in dpf.columns:
            if c.lower() in ("player_name", "full_name", "name"):
                name_col = c
                break
        if name_col:
            dpf["_canon_name"] = dpf[name_col].astype(str).apply(_canon)
            for side, pcol in (("away", "away_pitcher"), ("home", "home_pitcher")):
                raw_name = header.get(pcol)
                canon = _canon(raw_name)
                if canon:
                    prow = dpf[dpf["_canon_name"] == canon]
                    if not prow.empty:
                        pitchers[side] = _rename_pitcher_fields(prow.iloc[0].to_dict())

    # --------------------- Batter / Team Aggregates ------------------------
    bat_path = _latest_on_or_before(processed_dir, "batter_stat_features", game_date)
    bat = {}
    if bat_path:
        db = pd.read_csv(bat_path)
        if "team" in db.columns:
            db["team_n"] = db["team"].astype(str).apply(normalize_team)

            def build_team_bat_view(team_n: str):
                rowb = db[db["team_n"] == team_n]
                if rowb.empty:
                    return None
                r = rowb.iloc[0].to_dict()
                avg_launch_speed = _first(r, [
                    "avg_launch_speed", "avg_ev", "avg_exit_velocity", "avg_bat_speed"
                ])
                pa = r.get("PA") or r.get("plate_appearances")
                hr = r.get("HR") or r.get("recent_home_runs")
                so = r.get("SO") or r.get("recent_strikeouts")
                def rate(n, d):
                    try:
                        n = float(n); d = float(d)
                        return (n / d) if d and d > 0 else None
                    except Exception:
                        return None
                hr_rate = _pct01(_first(r, ["hr_rate", "hr_pct", "home_run_rate"])) or rate(hr, pa)
                k_rate  = _pct01(_first(r, ["k_rate", "k_pct", "strikeout_rate"])) or rate(so, pa)
                bb_rate = _pct01(_first(r, ["bb_rate", "bb_pct", "walk_rate"]))
                xwoba   = _first(r, ["xwoba", "avg_xwoba", "team_xwoba", "xwOBA"])
                return {
                    "avg_launch_speed": avg_launch_speed,
                    "hr_rate": hr_rate,
                    "k_rate": k_rate,
                    "bb_rate": bb_rate,
                    "xwoba": xwoba,
                    "_raw": r,
                }

            bat[away_n] = build_team_bat_view(away_n)
            bat[home_n] = build_team_bat_view(home_n)

    # -------------------- Pull main_features row (for summary) -------------
    mf_row = None
    mf_path = processed_dir / f"main_features_{date_str}.csv"
    if mf_path.exists():
        try:
            df_mf = pd.read_csv(mf_path)
            if "away_team" in df_mf.columns and "home_team" in df_mf.columns:
                df_mf["away_n"] = df_mf["away_team"].astype(str).apply(normalize_team)
                df_mf["home_n"] = df_mf["home_team"].astype(str).apply(normalize_team)
                mr = df_mf[(df_mf["away_n"] == away_n) & (df_mf["home_n"] == home_n)]
                if not mr.empty:
                    mf_row = mr.iloc[0].to_dict()
        except Exception:
            mf_row = None

    # ------------------------ Gemini Summary (uses MF) ---------------------
    summary = None
    try:
        context = {
            "date": date_str,
            "away_team": away_n,
            "home_team": home_n,
            "starting_pitchers": {
                "away": header.get("away_pitcher"),
                "home": header.get("home_pitcher"),
            },
            "prediction": prediction,
            "form": form,
            "pitchers": pitchers,
            "bat": bat,
            "result": result,
            "main_features": mf_row,   # feed MF row directly
        }
        summary = summarize_game(context)
    except Exception:
        summary = None

    # ---------------------------- Render -----------------------------------
    return render_template(
        "game_detail.html",
        date_str=date_str,
        away=away_n, home=home_n,
        header=header,
        prediction=prediction,
        result=result,
        form=form,
        pitchers=pitchers,
        bat=bat,
        summary=summary,
        away_logo=TEAM_LOGO_SLUG(away_n),
        home_logo=TEAM_LOGO_SLUG(home_n),
    )


# ------------------- Game Lookup (kept; points to new Hub) ------------------

@ui_bp.get("/game-lookup")
def game_lookup():
    """
    Show a searchable list of games built from main_features_*.csv.
    (Template links rows to /game/<date>/<away>-at-<home>)
    """
    processed_dir = Path(current_app.config["PROCESSED_DATA_DIR"])
    games: list[dict] = []

    for path in sorted(processed_dir.glob("main_features_*.csv")):
        try:
            df = pd.read_csv(path, usecols=["game_date", "home_team", "away_team"])
        except Exception:
            continue
        for row in df.to_dict(orient="records"):
            games.append(
                {
                    "game_date": row["game_date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                }
            )

    games.sort(key=lambda g: g["game_date"], reverse=True)
    return render_template("game_lookup.html", games=games)

# -------- Legacy row-index route -> redirect to new Game Hub ----------------

@ui_bp.get("/games/<game_date>/<int:row_index>")
def legacy_game_detail_redirect(game_date: str, row_index: int):
    """
    Legacy route kept for backward compatibility. Redirects to the new Game Hub
    using the teams from main_features_{game_date}.csv.
    """
    processed_dir = Path(current_app.config["PROCESSED_DATA_DIR"])
    csv_path = processed_dir / f"main_features_{game_date}.csv"
    if not csv_path.exists():
        return render_template("empty.html", title="Game", message=f"No main_features file for {game_date}."), 404

    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        return render_template("empty.html", title="Game", message="Invalid row index."), 400

    r = df.iloc[row_index].to_dict()
    away = normalize_team(r.get("away_team"))
    home = normalize_team(r.get("home_team"))
    return redirect(url_for("ui.game_detail", date_str=game_date, away=away, home=home), code=302)

# ----------------------------- Head-to-Head --------------------------------

@ui_bp.get("/head-to-head")
def h2h_lookup():
    """
    Head-to-Head lookup page:
    - Lists historical results (from historical_results.csv)
    - Client-side search (date/team/matchup)
    """
    try:
        rows = list_results_for_lookup(current_app.config["PROCESSED_DATA_DIR"])
    except FileNotFoundError:
        rows = []
    return render_template("h2h_lookup.html", rows=rows)

@ui_bp.get("/head-to-head/detail")
def h2h_detail():
    """
    Detail page for a selected matchup instance:
    ?date=YYYY-MM-DD&home=Dodgers&away=Padres
    Shows prior meetings with the same home/away orientation BEFORE that date.
    """
    date_str = request.args.get("date")
    home = request.args.get("home")
    away = request.args.get("away")

    if not (date_str and home and away):
        return render_template(
            "h2h_detail.html",
            error="Missing required parameters: date, home, away.",
            h2h=None, date=date_str, home=home, away=away
        ), 400

    h2h = get_prior_home_outcomes(
        current_app.config["PROCESSED_DATA_DIR"],
        game_date=date_str,
        home_team=home,
        away_team=away,
        last_n=10,
    )

    return render_template(
        "h2h_detail.html",
        error=None,
        h2h=h2h,
        date=date_str,
        home=home,
        away=away,
    )


# ----------------------------- Todays Matchuops ----------------------------

@ui_bp.get("/today")
def today_with_ev():
    """
    Combined 'Today's Games + EV vs Vegas' page.

    Expects:
      - processed/main_features_{date}.csv (required)
      - processed/readable_win_predictions_for_{date}_using_*.csv (optional)
      - processed/odds_{date}.csv (optional; schema below)

    odds_{date}.csv expected columns (example):
      game_date, away_team, home_team, sportsbook, away_ml, home_ml, fetched_at
    where *_ml are American moneyline ints (e.g., -135, +120).
    """
    processed_dir = Path(current_app.config["PROCESSED_DATA_DIR"])

    # -------- pick date (default to newest main_features_*) --------
    req_date = request.args.get("date")
    if req_date:
        date_str = req_date
    else:
        mf_files = sorted(processed_dir.glob("main_features_*.csv"))
        if not mf_files:
            return render_template(
                "empty.html",
                title="Today + EV",
                message="No main_features files found yet."
            ), 404
        # Get the last file and extract YYYY-MM-DD from its name
        m = re.search(r"(\d{4}-\d{2}-\d{2})", mf_files[-1].name)
        if not m:
            return render_template(
                "empty.html",
                title="Today + EV",
                message="Could not infer a date from main_features filenames."
            ), 500
        date_str = m.group(1)

    # ------------ Helpers ------------
    def implied_prob_from_ml(ml):
        """Return probability in [0,1] from American moneyline."""
        try:
            ml = int(ml)
        except Exception:
            return None
        if ml < 0:
            return (-ml) / ((-ml) + 100)     # e.g., -150 -> 150/(150+100)=0.6
        else:
            return 100 / (ml + 100)          # e.g., +120 -> 100/(120+100)=0.4545

    def profit_per_dollar_from_ml(ml):
        """Net profit per $1 stake (stake excluded)."""
        try:
            ml = int(ml)
        except Exception:
            return None
        if ml < 0:
            return 100 / (-ml)               # -150 -> 100/150=0.6667
        else:
            return ml / 100                  # +120 -> 1.2

    def expected_value_per_dollar(p, ml):
        """EV = p*profit - (1-p)*1; returns EV per $1 staked."""
        if p is None:
            return None
        prof = profit_per_dollar_from_ml(ml)
        if prof is None:
            return None
        return p * prof - (1 - p)

    # ---------- load main_features (anchor) ----------
    mf_path = processed_dir / f"main_features_{date_str}.csv"
    if not mf_path.exists():
        return render_template("empty.html", title="Today + EV",
                               message=f"main_features_{date_str}.csv not found."), 404
    df = pd.read_csv(mf_path)

    need_cols = {"away_team", "home_team"}
    if not need_cols.issubset(df.columns):
        return render_template("empty.html", title="Today + EV",
                               message="main_features is missing away_team/home_team."), 500

    df["away_team"] = df["away_team"].astype(str).apply(normalize_team)
    df["home_team"] = df["home_team"].astype(str).apply(normalize_team)
    if "game_date" not in df.columns:
        df["game_date"] = date_str

    # ---------- join predictions (optional) ----------
    pred_file = None
    pc = list(processed_dir.glob(f"readable_win_predictions_for_{date_str}_using_*.csv"))
    if pc:
        pred_file = pc[0]
        dp = pd.read_csv(pred_file)
        need = {"Game Date", "Home Team", "Away Team", "Win Probability", "Prediction"}
        if need.issubset(dp.columns):
            dp = dp.rename(columns={
                "Game Date": "game_date",
                "Home Team": "home_team",
                "Away Team": "away_team",
                "Win Probability": "home_win_prob",
                "Prediction": "model_pick"
            })
            dp["home_team"] = dp["home_team"].astype(str).apply(normalize_team)
            dp["away_team"] = dp["away_team"].astype(str).apply(normalize_team)
            df = df.merge(
                dp[["game_date", "home_team", "away_team", "home_win_prob", "model_pick"]],
                on=["game_date", "home_team", "away_team"],
                how="left",
            )

    # ---------- join odds (optional) ----------
    odds_path = processed_dir / f"odds_{date_str}.csv"
    if odds_path.exists():
        do = pd.read_csv(odds_path)
        for col in ("home_team", "away_team"):
            if col in do.columns:
                do[col] = do[col].astype(str).apply(normalize_team)
        # keep the first row per matchup (or customize for a specific book)
        keep_cols = {"home_team", "away_team", "away_ml", "home_ml", "sportsbook", "fetched_at"}
        do = do[[c for c in do.columns if c in keep_cols]].drop_duplicates(["home_team", "away_team"])
        # ensure *_ml are numeric
        for c in ("home_ml", "away_ml"):
            if c in do.columns:
                do[c] = pd.to_numeric(do[c], errors="coerce")
        df = df.merge(do, on=["home_team", "away_team"], how="left")
    else:
        # template safety
        df["away_ml"] = None
        df["home_ml"] = None
        df["sportsbook"] = None
        df["fetched_at"] = None

    # ---------- compute implied & EV ----------
    df["home_implied"] = df["home_ml"].apply(implied_prob_from_ml)
    df["away_implied"] = df["away_ml"].apply(implied_prob_from_ml)

    if "home_win_prob" not in df.columns:
        df["home_win_prob"] = None
    df["away_win_prob_model"] = df["home_win_prob"].apply(lambda p: (1 - p) if pd.notna(p) else None)

    df["ev_home"] = df.apply(lambda r: expected_value_per_dollar(r["home_win_prob"], r["home_ml"]), axis=1)
    df["ev_away"] = df.apply(lambda r: expected_value_per_dollar(r["away_win_prob_model"], r["away_ml"]), axis=1)

    def best_ev_row(r):
        eh, ea = r.get("ev_home"), r.get("ev_away")
        if eh is None and ea is None:
            return (None, None)
        if eh is None:
            return ("AWAY", ea)
        if ea is None:
            return ("HOME", eh)
        return ("HOME", eh) if eh >= ea else ("AWAY", ea)

    best_sides = df.apply(best_ev_row, axis=1, result_type="expand")
    df["best_side"] = best_sides[0]
    df["best_ev"] = best_sides[1]
    df["ev_flag"] = df["best_ev"].apply(lambda x: "+EV" if (pd.notna(x) and x > 0) else None)

    # sort: +EV first, then by EV value desc
    df = df.sort_values(["ev_flag", "best_ev"], ascending=[False, False], na_position="last")

    # ---------- logos & links ----------
    df["home_logo"] = df["home_team"].apply(TEAM_LOGO_SLUG)
    df["away_logo"] = df["away_team"].apply(TEAM_LOGO_SLUG)

    # ---------- compact Gemini blurb per row (safe no-op if no key) ----------
    def _blurb(row):
        try:
            return summarize_game_compact(row.to_dict())
        except Exception:
            return ""
    df["gemini_blurb"] = df.apply(_blurb, axis=1)

    rows = df.to_dict(orient="records")
    return render_template(
        "today_ev.html",
        date_str=date_str,
        rows=rows,
        has_predictions=pred_file is not None,
        has_odds=odds_path.exists(),
    )