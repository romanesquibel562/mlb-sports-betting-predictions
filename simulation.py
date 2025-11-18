# data_source/simulation.py

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Optional
# --- FIX: needed for canonical name cleaning
import re
# --- END FIX

# ---------- project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REF_DIR = PROJECT_ROOT / "utils" / "data" / "reference"

# ---------- file name formats ----------
PLAYER_FEATURES_FMT   = "player_features_{d}.csv"
PITCHER_FEATURES_FMT  = "pitcher_stat_features_{d}.csv"
BATTER_FEATURES_FMT   = "batter_stat_features_{d}.csv"
TEAM_FORM_FMT         = "team_form_{d}.csv"
MAIN_FEATURES_FMT     = "main_features_{d}.csv"

TEAM_ALIAS_FILE = REF_DIR / "team_aliases.csv"
DATE_COL_CANDIDATES = ("game_date", "event_date", "date")


@dataclass
class LoadedData:
    main_features: Optional[pd.DataFrame]
    matchups: Optional[pd.DataFrame]
    pitcher_stats: Optional[pd.DataFrame]
    batter_stats: Optional[pd.DataFrame]
    team_form: Optional[pd.DataFrame]
    source_files: Dict[str, str]


class SimulationDataSource:
    """
    Load historical artifacts for a requested game_date and (optionally) re-date them
    so downstream code treats them as 'today'.

    Priority:
      1) If main_features_{game_date}.csv exists → load (+ optional redate) and return.
      2) Else load nearest component files (<= game_date) and return them.
    """
    def __init__(self, game_date: date, redate: bool = True):
        self.game_date = game_date
        self.redate = redate
        self.team_alias_map = self._load_team_alias_map()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def load(self) -> LoadedData:
        sources: Dict[str, str] = {}

        # 1) try exact main_features first
        mf_path = self._path_for(MAIN_FEATURES_FMT, self.game_date)
        if mf_path.exists():
            mf = self._read_csv(mf_path)
            if self.redate:
                mf = self._redate(mf)
            # --- FIX: normalize teams even on the main-features happy path
            mf = self._normalize_teams_if_present(mf)
            # --- END FIX
            sources["main_features"] = str(mf_path)
            return LoadedData(
                main_features=mf,
                matchups=None,
                pitcher_stats=None,
                batter_stats=None,
                team_form=None,
                source_files=sources,
            )

        # 2) fallback: load nearest components
        pitcher_stats, p_path = self._load_nearest(PITCHER_FEATURES_FMT)
        batter_stats, b_path = self._load_nearest(BATTER_FEATURES_FMT)
        player_features, pl_path = self._load_nearest(PLAYER_FEATURES_FMT)  # not returned, but track source
        team_form, tf_path = self._load_nearest(TEAM_FORM_FMT)
        matchups, m_path = self._derive_matchups_from_nearest_main()

        if p_path:  sources["pitcher_stats"]   = str(p_path)
        if b_path:  sources["batter_stats"]    = str(b_path)
        if pl_path: sources["player_features"] = str(pl_path)
        if tf_path: sources["team_form"]       = str(tf_path)
        if m_path:  sources["matchups"]        = str(m_path)

        # redate everything we actually found
        if self.redate and pitcher_stats is not None:
            pitcher_stats = self._redate(pitcher_stats)
        if self.redate and batter_stats is not None:
            batter_stats = self._redate(batter_stats)
        if self.redate and team_form is not None:
            team_form = self._redate(team_form)
        if self.redate and matchups is not None:
            matchups = self._redate(matchups)

        # normalize teams on the fallback stuff
        if team_form is not None:
            team_form = self._normalize_team_form(team_form)
        if matchups is not None:
            matchups = self._normalize_teams_if_present(matchups)

        return LoadedData(
            main_features=None,
            matchups=matchups,
            pitcher_stats=pitcher_stats,
            batter_stats=batter_stats,
            team_form=team_form,
            source_files=sources,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_nearest(self, fmt: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        path, _ = self._nearest_on_or_before(fmt, self.game_date)
        if not path:
            return None, None
        df = self._read_csv(path)
        return df, path

    def _derive_matchups_from_nearest_main(self) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        mf_path, _ = self._nearest_on_or_before(MAIN_FEATURES_FMT, self.game_date)
        if not mf_path:
            return None, None
        df = self._read_csv(mf_path)
        keep = [
            c
            for c in (
                "game_date",
                "event_date",
                "away_team",
                "home_team",
                "away_pitcher",
                "home_pitcher",
            )
            if c in df.columns
        ]
        m = df[keep].copy() if keep else df.copy()
        return m, mf_path

    def _redate(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in DATE_COL_CANDIDATES:
            if col in df.columns:
                df[col] = pd.to_datetime(self.game_date)
        return df

    # --- FIX: central canonical cleaner
    def _clean_team_key(self, s: pd.Series) -> pd.Series:
        """
        Canonicalize team strings:
        - upper
        - strip
        - drop periods/apostrophes
        - collapse spaces
        - normalize common weird ones (ST. LOUIS -> ST LOUIS)
        """
        up = s.astype(str).str.upper().str.strip()
        up = up.str.replace(r"[.'’]", "", regex=True)
        up = up.str.replace(r"\s+", " ", regex=True)
        up = up.str.replace(r"\bST\s+LOUIS\b", "ST LOUIS", regex=True)
        up = up.str.replace(r"\bD ?BACKS\b", "DIAMONDBACKS", regex=True)
        up = up.str.replace(r"\bLA ANGELS\b", "LOS ANGELES ANGELS", regex=True)
        return up
    # --- END FIX

    def _normalize_team_form(self, df: pd.DataFrame) -> pd.DataFrame:
        # team_form usually has 'team' column
        if "team" in df.columns:
            up = self._clean_team_key(df["team"])
            df["team_std"] = (
                up.map(self.team_alias_map)
                  .fillna(up)
                  .str.strip()          # --- FIX: kill stray whitespace/newlines
            )
        return df

    def _normalize_teams_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add *_std columns using alias map; keep originals intact
        for col in ("home_team", "away_team", "team_name", "team"):
            if col in df.columns:
                up = self._clean_team_key(df[col])
                df[col + "_std"] = (
                    up.map(self.team_alias_map)
                      .fillna(up)
                      .str.strip()      # --- FIX: kill stray whitespace/newlines
                )
        return df

    def _read_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8")
        # --- FIX: auto-coerce date columns for consistency
        for col in DATE_COL_CANDIDATES:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # --- END FIX
        return df

    def _path_for(self, fmt: str, d: date) -> Path:
        return DATA_DIR / fmt.format(d=d.strftime("%Y-%m-%d"))

    def _nearest_on_or_before(self, fmt: str, target: date) -> Tuple[Optional[Path], Optional[date]]:
        prefix = fmt.split("{d}")[0]
        suffix = ".csv"
        candidates = sorted(DATA_DIR.glob(prefix + "*" + suffix))
        best_path, best_dt = None, None
        for p in candidates:
            stem = p.stem  # e.g. pitcher_stat_features_2025-10-27
            try:
                dt = datetime.strptime(stem.split("_")[-1], "%Y-%m-%d").date()
            except Exception:
                continue
            if dt <= target and (best_dt is None or dt > best_dt):
                best_path, best_dt = p, dt
        return best_path, best_dt

    def _load_team_alias_map(self) -> Dict[str, str]:
        # If alias file exists, read it, and normalize keys the same way we normalize inputs
        if TEAM_ALIAS_FILE.exists():
            m = pd.read_csv(TEAM_ALIAS_FILE)
            m["team_alias"] = self._clean_team_key(m["team_alias"])
            m["team_std"]   = self._clean_team_key(m["team_std"])
            return dict(zip(m["team_alias"], m["team_std"]))

        # --- FIX: comprehensive default alias map
        full_to_code = {
            # AL East
            "BOSTON RED SOX": "BOS",
            "NEW YORK YANKEES": "NYY",
            "TORONTO BLUE JAYS": "TOR",
            "TAMPA BAY RAYS": "TB",
            "BALTIMORE ORIOLES": "BAL",
            # AL Central
            "DETROIT TIGERS": "DET",
            "CLEVELAND GUARDIANS": "CLE",
            "MINNESOTA TWINS": "MIN",
            "KANSAS CITY ROYALS": "KC",
            "CHICAGO WHITE SOX": "CWS",
            # AL West
            "HOUSTON ASTROS": "HOU",
            "TEXAS RANGERS": "TEX",
            "SEATTLE MARINERS": "SEA",
            "LOS ANGELES ANGELS": "LAA",
            "OAKLAND ATHLETICS": "OAK",
            # NL East
            "ATLANTA BRAVES": "ATL",
            "MIAMI MARLINS": "MIA",
            "NEW YORK METS": "NYM",
            "PHILADELPHIA PHILLIES": "PHI",
            "WASHINGTON NATIONALS": "WSH",
            # NL Central
            "CHICAGO CUBS": "CHC",
            "CINCINNATI REDS": "CIN",
            "MILWAUKEE BREWERS": "MIL",
            "PITTSBURGH PIRATES": "PIT",
            "ST LOUIS CARDINALS": "STL",   # cleaned (no period)
            # NL West
            "LOS ANGELES DODGERS": "LAD",
            "SAN DIEGO PADRES": "SD",
            "SAN FRANCISCO GIANTS": "SF",
            "ARIZONA DIAMONDBACKS": "ARI",
            "COLORADO ROCKIES": "COL",
        }

        extras = {
            "DODGERS": "LAD",
            "ANGELS": "LAA",
            "ATHLETICS": "OAK",
            "A S": "OAK",  # "A's" -> "A S"
            "GIANTS": "SF",
            "PADRES": "SD",
            "DIAMONDBACKS": "ARI",
            "METS": "NYM",
            "YANKEES": "NYY",
            "RED SOX": "BOS",
            "WHITE SOX": "CWS",
            "CUBS": "CHC",
            "REDS": "CIN",
            "BREWERS": "MIL",
            "PIRATES": "PIT",
            "CARDINALS": "STL",
            "PHILLIES": "PHI",
            "BRAVES": "ATL",
            "MARLINS": "MIA",
            "NATIONALS": "WSH",
            "GUARDIANS": "CLE",
            "RANGERS": "TEX",
            "ASTROS": "HOU",
            "MARINERS": "SEA",
            "ROYALS": "KC",
            "TWINS": "MIN",
            "TIGERS": "DET",
            "ORIOLES": "BAL",
            "RAYS": "TB",
            "BLUE JAYS": "TOR",
            "ROCKIES": "COL",
        }

        # pass-through codes
        codes = {
            "BOS": "BOS", "NYY": "NYY", "TOR": "TOR", "TB": "TB", "BAL": "BAL",
            "DET": "DET", "CLE": "CLE", "MIN": "MIN", "KC": "KC", "CWS": "CWS",
            "HOU": "HOU", "TEX": "TEX", "SEA": "SEA", "LAA": "LAA", "OAK": "OAK",
            "ATL": "ATL", "MIA": "MIA", "NYM": "NYM", "PHI": "PHI", "WSH": "WSH",
            "CHC": "CHC", "CIN": "CIN", "MIL": "MIL", "PIT": "PIT", "STL": "STL",
            "LAD": "LAD", "SD": "SD", "SF": "SF", "ARI": "ARI", "COL": "COL",
        }

        alias_map: Dict[str, str] = {}
        alias_map.update(full_to_code)
        alias_map["ST. LOUIS CARDINALS"] = "STL"  # dotted version too
        alias_map.update(extras)
        alias_map.update(codes)

        # clean all keys to match _clean_team_key
        cleaned_map: Dict[str, str] = {}
        for k, v in alias_map.items():
            kk = re.sub(r"[.'’]", "", k).upper().strip()
            kk = re.sub(r"\s+", " ", kk)
            cleaned_map[kk] = v
        cleaned_map["ST LOUIS CARDINALS"] = "STL"

        return cleaned_map
        # --- END FIX


# ---------------------- Self-test (optional) ----------------------
if __name__ == "__main__":
    # change date to test different days
    test_date = date(2025, 9, 20)
    sim = SimulationDataSource(game_date=test_date, redate=True)
    loaded = sim.load()

    print("== Simulation load ==")
    print("Project root:", PROJECT_ROOT)
    for k, v in loaded.source_files.items():
        print(f"{k:14s} -> {v}")

    if loaded.main_features is not None:
        print("\nmain_features head:")
        print(loaded.main_features.head())
    else:
        if loaded.matchups is not None:
            print("\nmatchups head:")
            print(loaded.matchups.head())
        if loaded.pitcher_stats is not None:
            print("\npitcher_stats head:")
            print(loaded.pitcher_stats.head())
        if loaded.batter_stats is not None:
            print("\nbatter_stats head:")
            print(loaded.batter_stats.head())
        if loaded.team_form is not None:
            print("\nteam_form head:")
            print(loaded.team_form.head())

# run code:
# cd C:\Users\roman\baseball_forecast_project\data_source
# python simulation.py
