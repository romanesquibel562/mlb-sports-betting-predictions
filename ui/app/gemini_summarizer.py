# ui/app/gemini_summarizer.py

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if x < 1.0:  # treat 0.62 as 62%
            x *= 100.0
        return f"{x:.1f}%"
    except Exception:
        return "—"

def _fmt(x, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def _safe_div(n, d) -> Optional[float]:
    try:
        n = float(n); d = float(d)
        if d == 0:
            return None
        return n / d
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Context builder (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def build_game_context(game_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts a dict (e.g., main_features row merged with predictions/odds)
    and returns a normalized context for LLM prompts.
    """
    metadata = {
        "game_date": game_row.get("game_date"),
        "home_team": game_row.get("home_team"),
        "away_team": game_row.get("away_team"),
        "home_pitcher": game_row.get("home_pitcher"),
        "away_pitcher": game_row.get("away_pitcher"),
    }

    pitchers = {
        "home": {
            "name": game_row.get("home_pitcher"),
            "total_pitches": game_row.get("home_pitcher_total_pitches"),
            "avg_velocity": game_row.get("home_pitcher_avg_velocity"),
            "avg_spin_rate": game_row.get("home_pitcher_avg_spin_rate"),
            "avg_extension": game_row.get("home_pitcher_avg_extension"),
            "strikeouts": game_row.get("home_pitcher_strikeouts"),
            "whiffs": game_row.get("home_pitcher_whiffs"),
            "avg_bat_speed_allowed": game_row.get("home_pitcher_avg_bat_speed"),
            "avg_launch_angle_allowed": game_row.get("home_pitcher_avg_launch_angle"),
            "avg_exit_velocity_allowed": game_row.get("home_pitcher_avg_exit_velocity"),
            "avg_swing_length_allowed": game_row.get("home_pitcher_avg_swing_length"),
            "games_sampled": game_row.get("home_pitcher_games_played"),
        },
        "away": {
            "name": game_row.get("away_pitcher"),
            "total_pitches": game_row.get("away_pitcher_total_pitches"),
            "avg_velocity": game_row.get("away_pitcher_avg_velocity"),
            "avg_spin_rate": game_row.get("away_pitcher_avg_spin_rate"),
            "avg_extension": game_row.get("away_pitcher_avg_extension"),
            "strikeouts": game_row.get("away_pitcher_strikeouts"),
            "whiffs": game_row.get("away_pitcher_whiffs"),
            "avg_bat_speed_allowed": game_row.get("away_pitcher_avg_bat_speed"),
            "avg_launch_angle_allowed": game_row.get("away_pitcher_avg_launch_angle"),
            "avg_exit_velocity_allowed": game_row.get("away_pitcher_avg_exit_velocity"),
            "avg_swing_length_allowed": game_row.get("away_pitcher_avg_swing_length"),
            "games_sampled": game_row.get("away_pitcher_games_played"),
        },
    }

    batters = {
        "home_team_offense": {
            "avg_launch_speed": game_row.get("home_team_avg_launch_speed"),
            "avg_bat_speed": game_row.get("home_team_avg_bat_speed"),
            "avg_swing_length": game_row.get("home_team_avg_swing_length"),
        },
        "away_team_offense": {
            "avg_launch_speed": game_row.get("away_team_avg_launch_speed"),
            "avg_bat_speed": game_row.get("away_team_avg_bat_speed"),
            "avg_swing_length": game_row.get("away_team_avg_swing_length"),
        },
    }

    team_form = {
        "home": {
            "wins": game_row.get("home_wins"),
            "losses": game_row.get("home_losses"),
            "run_diff": game_row.get("home_run_diff"),
            "streak": game_row.get("home_streak"),
            "games_played": game_row.get("home_games_played"),
            "win_pct": game_row.get("home_win_pct"),
        },
        "away": {
            "wins": game_row.get("away_wins"),
            "losses": game_row.get("away_losses"),
            "run_diff": game_row.get("away_run_diff"),
            "streak": game_row.get("away_streak"),
            "games_played": game_row.get("away_games_played"),
            "win_pct": game_row.get("away_win_pct"),
        },
    }

    prediction = {
        "home_win_probability": (
            game_row.get("home_win_probability")
            or game_row.get("win_probability")
            or game_row.get("home_win_prob")
        ),
        "model_pick": (
            game_row.get("Prediction")
            or game_row.get("model_pick")
            or game_row.get("pick")
        ),
    }

    odds = {
        "away_ml": game_row.get("away_ml"),
        "home_ml": game_row.get("home_ml"),
        "sportsbook": game_row.get("sportsbook"),
        "fetched_at": game_row.get("fetched_at"),
    }

    return {
        "metadata": metadata,
        "pitchers": pitchers,
        "batters": batters,
        "team_form": team_form,
        "prediction": prediction,
        "odds": odds,
    }

# ---------------------------------------------------------------------------
# Brief builder (deterministic, no-LLM)
# ---------------------------------------------------------------------------

def _build_brief_from_context(ctx: Dict[str, Any]) -> str:
    """
    Build a compact, data-aware text brief using ONLY fields present.
    Works with either a 'rich' context or a flat main_features-like row.
    """
    is_rich = any(k in ctx for k in ("main_features", "metadata", "away_team", "home_team"))

    if is_rich:
        meta = ctx.get("metadata", {})
        date = ctx.get("date") or ctx.get("game_date") or meta.get("game_date") or "—"
        away = ctx.get("away_team") or meta.get("away_team") or "Away"
        home = ctx.get("home_team") or meta.get("home_team") or "Home"

        mf   = ctx.get("main_features") or {}
        pred = ctx.get("prediction") or {}
        form = ctx.get("form") or {}
        result = ctx.get("result") or {}
        start_p = (ctx.get("starting_pitchers") or {})
        sp_away = mf.get("away_pitcher") or start_p.get("away")
        sp_home = mf.get("home_pitcher") or start_p.get("home")

        hp_velo   = mf.get("home_pitcher_avg_velocity")
        hp_spin   = mf.get("home_pitcher_avg_spin_rate")
        hp_whiff_rate = _safe_div(mf.get("home_pitcher_whiffs"), mf.get("home_pitcher_total_pitches"))

        ap_velo   = mf.get("away_pitcher_avg_velocity")
        ap_spin   = mf.get("away_pitcher_avg_spin_rate")
        ap_whiff_rate = _safe_div(mf.get("away_pitcher_whiffs"), mf.get("away_pitcher_total_pitches"))

        home_ev = mf.get("home_team_avg_launch_speed")
        away_ev = mf.get("away_team_avg_launch_speed")

        win_prob = pred.get("win_prob") or pred.get("home_win_probability") or pred.get("win_probability")
        pick     = pred.get("pick") or pred.get("model_pick") or pred.get("Prediction")

        def _form_line(team_name: str) -> str:
            f = form.get(team_name)
            if not f:
                return f"- {team_name}: (no recent form)\n"
            return (f"- {team_name}: W-L {f.get('wins','?')}-{f.get('losses','?')}, "
                    f"RunDiff {f.get('run_diff','?')}, Streak {f.get('streak','?')}, "
                    f"Win% {_fmt_pct(f.get('win_pct'))}\n")

        lines = []
        lines.append(f"{away} at {home} — {date}\n")

        if pick and win_prob is not None:
            lines.append(f"Model: {pick} {_fmt_pct(win_prob)} win probability.\n")

        lines.append("Starting Pitchers:\n")
        lines.append(f"- Away: {sp_away or '—'}")
        sub = []
        if ap_velo is not None: sub.append(f"Velo {_fmt(ap_velo)}")
        if ap_spin is not None: sub.append(f"Spin {_fmt(ap_spin)}")
        if ap_whiff_rate is not None: sub.append(f"Whiff% {_fmt_pct(ap_whiff_rate)}")
        if sub: lines[-1] += " | " + "; ".join(sub)
        lines[-1] += "\n"

        lines.append(f"- Home: {sp_home or '—'}")
        sub = []
        if hp_velo is not None: sub.append(f"Velo {_fmt(hp_velo)}")
        if hp_spin is not None: sub.append(f"Spin {_fmt(hp_spin)}")
        if hp_whiff_rate is not None: sub.append(f"Whiff% {_fmt_pct(hp_whiff_rate)}")
        if sub: lines[-1] += " | " + "; ".join(sub)
        lines[-1] += "\n"

        lines.append("Team Offense (main_features):\n")
        lines.append(f"- {away}: AvgEV {_fmt(away_ev)}\n")
        lines.append(f"- {home}: AvgEV {_fmt(home_ev)}\n")

        if form:
            lines.append("Recent Form:\n")
            lines.append(_form_line(away))
            lines.append(_form_line(home))

        if result.get("actual_winner"):
            lines.append(f"Final: Winner — {result.get('actual_winner')}.\n")

        brief = "".join(lines).strip()
        if not brief or brief == f"{away} at {home} — {date}":
            brief = (f"{away} at {home} — {date}\n"
                     "Limited context available. When predictions, pitcher metrics, "
                     "or team snapshots are present, a fuller summary will appear.")
        return brief

    # Legacy: treat ctx as a flat row
    row = ctx
    date = row.get("game_date") or "—"
    away = row.get("away_team") or "Away"
    home = row.get("home_team") or "Home"

    ap_velo = row.get("away_pitcher_avg_velocity")
    ap_spin = row.get("away_pitcher_avg_spin_rate")
    ap_whiff_rate = _safe_div(row.get("away_pitcher_whiffs"), row.get("away_pitcher_total_pitches"))

    hp_velo = row.get("home_pitcher_avg_velocity")
    hp_spin = row.get("home_pitcher_avg_spin_rate")
    hp_whiff_rate = _safe_div(row.get("home_pitcher_whiffs"), row.get("home_pitcher_total_pitches"))

    away_ev = row.get("away_team_avg_launch_speed")
    home_ev = row.get("home_team_avg_launch_speed")

    lines = []
    lines.append(f"{away} at {home} — {date}\n")
    lines.append("Starting Pitchers:\n")
    lines.append(f"- Away: {row.get('away_pitcher') or '—'}")
    sub = []
    if ap_velo is not None: sub.append(f"Velo {_fmt(ap_velo)}")
    if ap_spin is not None: sub.append(f"Spin {_fmt(ap_spin)}")
    if ap_whiff_rate is not None: sub.append(f"Whiff% {_fmt_pct(ap_whiff_rate)}")
    if sub: lines[-1] += " | " + "; ".join(sub)
    lines[-1] += "\n"

    lines.append(f"- Home: {row.get('home_pitcher') or '—'}")
    sub = []
    if hp_velo is not None: sub.append(f"Velo {_fmt(hp_velo)}")
    if hp_spin is not None: sub.append(f"Spin {_fmt(hp_spin)}")
    if hp_whiff_rate is not None: sub.append(f"Whiff% {_fmt_pct(hp_whiff_rate)}")
    if sub: lines[-1] += " | " + "; ".join(sub)
    lines[-1] += "\n"

    lines.append("Team Offense (main_features):\n")
    lines.append(f"- {away}: AvgEV {_fmt(away_ev)}\n")
    lines.append(f"- {home}: AvgEV {_fmt(home_ev)}\n")

    brief = "".join(lines).strip()
    if not brief or brief == f"{away} at {home} — {date}":
        brief = (f"{away} at {home} — {date}\n"
                 "Limited context available. When predictions, pitcher metrics, "
                 "or team snapshots are present, a fuller summary will appear.")
    return brief

# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def summarize_game(context_or_row: Dict[str, Any]) -> str:
    """
    Rich 3–6 sentence paragraph for the Game Detail page.
    Returns HTML (<p> or <pre>) so templates can render with |safe.
    """
    brief = _build_brief_from_context(context_or_row)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"<pre style='white-space:pre-wrap'>{brief}</pre>"

    try:
        genai.configure(api_key=api_key)
        sys_instr = (
            "You are an MLB analyst. Write a concise, neutral matchup brief (3–6 sentences). "
            "Use only the provided facts; do not fabricate stats."
        )
        prompt = (
            f"{sys_instr}\n\n"
            "Context (use only what's present; omit anything missing):\n"
            f"{brief}\n\n"
            "Output: a short paragraph. Mention probabilities once if present."
        )
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            text = brief
        return f"<p>{text}</p>"
    except Exception:
        return f"<pre style='white-space:pre-wrap'>{brief}</pre>"

def summarize_game_compact(game_row: Dict[str, Any]) -> str:
    """
    Ultra-compact 1–3 sentence blurb for list cards (Today + EV).
    Returns plain text (no HTML).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""  # silent no-op

    # Build a clean context (includes odds/model if present in row)
    ctx = build_game_context(game_row)
    ctx_json = json.dumps(ctx, indent=2, default=str)

    away = (ctx.get("metadata") or {}).get("away_team", "Away")
    home = (ctx.get("metadata") or {}).get("home_team", "Home")

    prompt = f"""
You write ultra-compact betting notes for MLB games.

Input JSON:
```json
{ctx_json}
```

requirements:
- Use 1–3 sentences.
- Focus on edges for betting (pitchers, model pick, win prob, odds), matchup strengths, EV signals, or team trends, and model vs odds edges.
- Mention teams by name (e.g., "{away}", "{home}").
- Absolutely no fabrication of stats.
- Output a single paragraph of plain text, no bullet points or lists.

Game: {away} at {home}.
""".strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        return ""
