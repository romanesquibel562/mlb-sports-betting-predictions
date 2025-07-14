# scrape_team_form_mlb.py

import requests
import pandas as pd
import logging
from pathlib import Path

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "team_recent_form.csv"

# === Team Normalization Map ===
TEAM_NAME_MAP = {
    'New York Yankees': 'NYY',
    'Boston Red Sox': 'BOS',
    'Tampa Bay Rays': 'TB',
    'Toronto Blue Jays': 'TOR',
    'Baltimore Orioles': 'BAL',
    'Cleveland Guardians': 'CLE',
    'Detroit Tigers': 'DET',
    'Kansas City Royals': 'KC',
    'Chicago White Sox': 'CWS',
    'Minnesota Twins': 'MIN',
    'Houston Astros': 'HOU',
    'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX',
    'Los Angeles Angels': 'LAA',
    'Oakland Athletics': 'OAK',
    'Atlanta Braves': 'ATL',
    'Miami Marlins': 'MIA',
    'New York Mets': 'NYM',
    'Philadelphia Phillies': 'PHI',
    'Washington Nationals': 'WSH',
    'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN',
    'Milwaukee Brewers': 'MIL',
    'Pittsburgh Pirates': 'PIT',
    'St. Louis Cardinals': 'STL',
    'Arizona Diamondbacks': 'ARI',
    'Colorado Rockies': 'COL',
    'Los Angeles Dodgers': 'LAD',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF'
}

def scrape_team_form_mlb():
    url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025&standingsTypes=regularSeason"
    response = requests.get(url)
    data = response.json()

    teams = []
    for record_type in data.get('records', []):
        for team_rec in record_type.get('teamRecords', []):
            team_name = team_rec['team'].get('name', 'Unknown')
            normalized_name = TEAM_NAME_MAP.get(team_name, team_name)
            teams.append({
                "team": normalized_name,
                "wins": team_rec.get('wins'),
                "losses": team_rec.get('losses'),
                "run_diff": team_rec.get('runDifferential'),
                "streak": team_rec.get('streak', {}).get('streakCode', ''),
                "games_played": team_rec.get('gamesPlayed'),
                "win_pct": team_rec.get('winningPercentage')
            })

    df = pd.DataFrame(teams)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved {len(df)} team rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    scrape_team_form_mlb()
    
# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrape_team_form_mlb.py
