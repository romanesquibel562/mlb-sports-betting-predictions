# generate_power_rankings.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import argparse
import re
import os
import datetime as dt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
UI_DATA = PROJECT_ROOT / "ui" / "data"
CONFIG_DEFAULT = PROJECT_ROOT / "config" / "ui_weights.yml"

def z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (x - mu) / sd

def find_latest_csv(pattern: str) -> Path | None:
    c = [p for p in DATA_PROCESSED.glob("*.csv") if re.search(pattern, p.name)]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None

def load_cfg(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "team_gpa_weights": {
            "recent_win_pct": 0.45,
            "run_diff_per_game": 0.25,
            "batting_power_idx": 0.30,
        },
        "batter_columns": {
            "team_file": {"team": "team_name", "avg_launch_speed": "avg_launch_speed", "hr_proxy": None},
            "player_file": {"team": "team_name", "avg_launch_speed": "avg_launch_speed", "hr_rate": "home_run_rate"},
        },
        "pitcher_columns": {"team": "team_name", "era": None, "k9": None, "whip": None},
        "data_windows": {"lookback_days_batting": 60},
    }

def normalize_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _parse_date_from_name(fname: str) -> dt.date | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", fname)
    if not m: 
        return None
    y, mo, d = map(int, m.groups())
    return dt.date(y, mo, d)

def _list_window_files(pattern: str, lookback_days: int) -> list[Path]:
    today = dt.date.today()
    start = today - dt.timedelta(days=int(lookback_days))
    files = []
    for p in DATA_PROCESSED.glob("*.csv"):
        if not re.search(pattern, p.name):
            continue
        d = _parse_date_from_name(p.name)
        if d and start <= d <= today:
            files.append((d, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]

def add_batting_idx(df_form: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b_cfg = cfg.get("batter_columns", {})
    lookback_days = cfg.get("data_windows", {}).get("lookback_days_batting", 60)

    team_files = _list_window_files(r"^team_batter_stats_\d{4}-\d{2}-\d{2}\.csv$", lookback_days)
    player_files = _list_window_files(r"^player_features_\d{4}-\d{2}-\d{2}\.csv$", lookback_days)

    team_frames = []
    if team_files:
        t_team = b_cfg.get("team_file", {})
        team_col = t_team.get("team", "team_name")
        ls_col   = t_team.get("avg_launch_speed", "avg_launch_speed")
        hr_proxy = t_team.get("hr_proxy", None)
        for f in team_files:
            tb = pd.read_csv(f)
            if team_col not in tb.columns:
                continue
            parts = []
            if ls_col in tb.columns: parts.append(pd.to_numeric(tb[ls_col], errors="coerce"))
            if hr_proxy and hr_proxy in tb.columns: parts.append(pd.to_numeric(tb[hr_proxy], errors="coerce"))
            if parts:
                df = pd.DataFrame({"_team_norm": normalize_team(tb[team_col])})
                df["bp_raw"] = parts[0] if len(parts) == 1 else pd.DataFrame(parts).T.mean(axis=1)
                team_frames.append(df)

    if player_files:
        p_team = b_cfg.get("player_file", {})
        team_col = p_team.get("team", "team_name")
        ls_col   = p_team.get("avg_launch_speed", "avg_launch_speed")
        hr_col   = p_team.get("hr_rate", "home_run_rate")
        for f in player_files:
            pf = pd.read_csv(f)
            if team_col not in pf.columns:
                continue
            agg_cols = {}
            if ls_col in pf.columns: agg_cols[ls_col] = "mean"
            if hr_col in pf.columns: agg_cols[hr_col] = "mean"
            if not agg_cols:
                continue
            agg = pf.groupby(team_col, as_index=False).agg(agg_cols)
            parts = []
            if ls_col in agg.columns: parts.append(pd.to_numeric(agg[ls_col], errors="coerce"))
            if hr_col in agg.columns: parts.append(pd.to_numeric(agg[hr_col], errors="coerce"))
            if parts:
                df = pd.DataFrame({"_team_norm": normalize_team(agg[team_col])})
                df["bp_raw"] = parts[0] if len(parts) == 1 else pd.DataFrame(parts).T.mean(axis=1)
                team_frames.append(df)

    if not team_frames:
        df_form["batting_power_idx"] = np.nan
        return df_form

    all_batting = pd.concat(team_frames, ignore_index=True)
    by_team = all_batting.groupby("_team_norm", as_index=False).agg({"bp_raw": "mean"})
    by_team["batting_power_idx"] = z(by_team["bp_raw"])

    df_form["_team_norm"] = normalize_team(df_form["team"])
    df_form = df_form.merge(by_team[["_team_norm", "batting_power_idx"]], on="_team_norm", how="left")
    df_form.drop(columns=["_team_norm"], inplace=True)
    return df_form

def add_pitching_idx(df_form: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Leave column absent unless you actually compute it in the future.
    return df_form

def build_rankings(cfg_path: Path, out_dir: Path, team_form_path: Path | None = None) -> Path | None:
    cfg = load_cfg(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    form_path = team_form_path or find_latest_csv(r"^team_form_\d{4}-\d{2}-\d{2}\.csv$")
    if not form_path or not form_path.exists():
        raise FileNotFoundError("Could not find team_form_YYYY-MM-DD.csv in data/processed.")
    rf = pd.read_csv(form_path)

    if "team" not in rf.columns:
        raise ValueError(f"{form_path.name} must include a 'team' column.")

    for c in ["wins","losses","run_diff","games_played","win_pct"]:
        if c in rf.columns:
            rf[c] = pd.to_numeric(rf[c], errors="coerce")

    rf["recent_win_pct"] = np.where(rf["games_played"].fillna(0) > 0, rf["wins"] / rf["games_played"], np.nan)
    rf["run_diff_per_game"] = np.where(rf["games_played"].fillna(0) > 0, rf["run_diff"] / rf["games_played"], np.nan)

    rf = add_batting_idx(rf, cfg)
    rf = add_pitching_idx(rf, cfg)

    rf["_z_recent_win_pct"] = z(rf["recent_win_pct"])
    rf["_z_run_diff_per_game"] = z(rf["run_diff_per_game"])

    pieces = {
        "recent_win_pct": rf["_z_recent_win_pct"],
        "run_diff_per_game": rf["_z_run_diff_per_game"],
        "batting_power_idx": rf["batting_power_idx"],
    }
    if "pitching_quality_idx" in rf.columns and rf["pitching_quality_idx"].notna().any():
        pieces["pitching_quality_idx"] = rf["pitching_quality_idx"]

    weights_cfg = cfg.get("team_gpa_weights", {})
    base_w = np.array([weights_cfg.get(k, 0.0) for k in pieces.keys()], dtype=float)

    comp_vals = np.column_stack([pieces[k].to_numpy() for k in pieces.keys()])
    comp_mask = np.isfinite(comp_vals)

    weights = np.tile(base_w, (comp_vals.shape[0], 1))
    weights[~comp_mask] = 0.0
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, np.where(row_sums == 0, 1.0, row_sums))

    rf["team_gpa_raw"] = np.nansum(np.where(comp_mask, comp_vals * weights, 0.0), axis=1)
    m, M = np.nanmin(rf["team_gpa_raw"]), np.nanmax(rf["team_gpa_raw"])
    rf["team_gpa"] = np.where(np.isfinite(m) & np.isfinite(M) & (M > m), 100 * (rf["team_gpa_raw"] - m) / (M - m), 50.0)

    cols = ["team", "team_gpa", "recent_win_pct", "run_diff_per_game", "batting_power_idx"]
    if "pitching_quality_idx" in pieces:
        cols.append("pitching_quality_idx")

    out = (rf[cols]
           .drop_duplicates(subset=["team"])
           .sort_values(by="team_gpa", ascending=False)
           .reset_index(drop=True))
    out.insert(0, "rank", out.index + 1)

    out["trend"] = 0
    prev_path = out_dir / "team_power_rankings.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, dtype={"team": str})
        if {"team", "rank"}.issubset(prev.columns):
            m = out[["team","rank"]].merge(prev[["team","rank"]].rename(columns={"rank":"rank_prev"}), on="team", how="left")
            out["trend"] = (m["rank_prev"] - m["rank"]).fillna(0).astype(int)

    out["team_gpa"] = out["team_gpa"].round(2)
    for c in ["recent_win_pct", "run_diff_per_game", "batting_power_idx"]:
        if c in out.columns:
            out[c] = out[c].round(3)
    if "pitching_quality_idx" in out.columns:
        out["pitching_quality_idx"] = out["pitching_quality_idx"].round(3)
    out["rank"] = out["rank"].astype(int)

    csv_path = out_dir / "team_power_rankings.csv"
    json_path = out_dir / "team_power_rankings.json"
    dated = out_dir / f"team_power_rankings_{dt.date.today().isoformat()}.csv"

    out.to_csv(csv_path, index=False)
    out.to_json(json_path, orient="records", indent=2)
    if not dated.exists():
        out.to_csv(dated, index=False)

    print(f"Saved:\n  {csv_path}\n  {json_path}\n  {dated}")
    return csv_path

def parse_args():
    ap = argparse.ArgumentParser(description="Generate Team Power Rankings for UI")
    ap.add_argument("--weights", type=str, default=str(CONFIG_DEFAULT), help="Path to ui_weights.yml")
    ap.add_argument("--outdir", type=str, default=str(UI_DATA), help="Output directory for UI artifacts")
    ap.add_argument("--team-form", type=str, help="Optional explicit path to team_form_YYYY-MM-DD.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    UI_DATA.mkdir(parents=True, exist_ok=True)
    build_rankings(cfg_path=Path(args.weights), out_dir=Path(args.outdir),
                   team_form_path=Path(args.team_form) if args.team_form else None)
    


# run
# cd C:\Users\roman\baseball_forecast_project\modeling
# python generate_power_rankings.py --weights ../config/ui_weights.yml --outdir ../ui/data
