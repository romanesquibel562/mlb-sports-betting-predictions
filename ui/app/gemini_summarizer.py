# ui/app/gemini_summarizer.py

import os
import json
import google.generativeai as genai
from typing import Dict, Any

MODEL_NAME = "gemini-2.5-flash"

def build_game_context(game_row: Dict[str, Any]) -> Dict[str, Any]:

    # Basic metadata
    metadata = {
        "game_date": game_row.get("game_date"),
        "home_team": game_row.get("home_team"),
        "away_team": game_row.get("away_team"),
        "home_pitcher": game_row.get("home_pitcher"),
        "away_pitcher": game_row.get("away_pitcher"),
    }

    # Pitcher stat blocks (30-day / recent performance)
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

    # Team-level batter quality (from your Statcast aggregation)
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

    # Recent form / standings style info
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

    # Optional: model prediction info, if present in the row (e.g. joined from readable_win_predictions)
    prediction = {
        # e.g. 0.62
        "home_win_probability": game_row.get("home_win_probability") or game_row.get("win_probability"),
        # e.g. "Pick: BAL" or "BAL"
        "model_pick": game_row.get("Prediction") or game_row.get("model_pick"),
    }

    context = {
        "metadata": metadata,
        "pitchers": pitchers,
        "batters": batters,
        "team_form": team_form,
        "prediction": prediction,
    }
    return context


def summarize_game(game_row: Dict[str, Any]) -> str:
    """
    Given a dict representing ONE ROW from your merged matchup/features/predictions
    dataset, call Gemini and return a human-readable analytical summary.

    This function does NOT read CSVs – it just shapes the data and prompts Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    game_context = build_game_context(game_row)
    context_json = json.dumps(game_context, indent=2, default=str)

    prompt = f"""
You are an advanced MLB analyst writing for serious baseball fans and bettors.

You are given a JSON object with:
- "metadata": basic info (teams, date, pitchers)
- "pitchers": recent Statcast-based pitching metrics for home & away
- "batters": team-level offensive quality metrics for home & away
- "team_form": recent wins/losses, streaks, run differential, win%
- "prediction": the model's win probability and pick (if available)

Here is the game context:

```json
{context_json}
```
Using ONLY the data provided in the JSON above, write a concise analytical summary of the upcoming game between {game_row.get("away_team")} and {game_row.get("home_team")}. Focus on key insights from the pitcher and batter metrics, recent team form, and any
model predictions. Avoid speculation or information not in the JSON.
Provide the summary in 3-5 sentences, suitable for publication on a baseball analysis site.
    """.strip()

    response = model.generate_content(prompt)

    return response.text