# generate_historical_team_form.py

import os
import glob
import logging
import pandas as pd
import requests
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Team name mapping
TEAM_NAME_MAP = {
    'New York Yankees': 'NYY', 'Boston Red Sox': 'BOS', 'Tampa Bay Rays': 'TB',
    'Toronto Blue Jays': 'TOR', 'Baltimore Orioles': 'BAL', 'Cleveland Guardians': 'CLE',
    'Detroit Tigers': 'DET', 'Kansas City Royals': 'KC', 'Chicago White Sox': 'CWS',
    'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU', 'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX', 'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK',
    'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM',
    'Philadelphia Phillies': 'PHI', 'Washington Nationals': 'WSH', 'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL', 'Pittsburgh Pirates': 'PIT',
    'St. Louis Cardinals': 'STL', 'Arizona Diamondbacks': 'ARI', 'Colorado Rockies': 'COL',
    'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF'
}

# Paths
results_dir = "C:/Users/roman/baseball_forecast_project/data/processed"
output_dir = results_dir

def scrape_team_form_api_for_date(date_str):
    """Scrape standings from MLB API and save as team_form_<date>.csv."""
    year = datetime.strptime(date_str, "%Y-%m-%d").year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}&standingsTypes=regularSeason&date={date_str}"
    
    try:
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
        output_path = os.path.join(output_dir, f"team_form_{date_str}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved team form for {date_str} to {output_path}")
    except Exception as e:
        logger.error(f"Failed to scrape team form for {date_str}: {e}")

def run_team_form_rolling():
    existing = {os.path.basename(f).split("_")[-1].replace(".csv", "") for f in 
                glob.glob(os.path.join(output_dir, "team_form_*.csv"))}
    results_files = sorted(glob.glob(os.path.join(results_dir, "historical_results_*.csv")))

    for f in results_files:
        date_str = os.path.basename(f).split("_")[-1].replace(".csv", "")
        if date_str in existing:
            continue
        scrape_team_form_api_for_date(date_str)

if __name__ == "__main__":
    run_team_form_rolling()

    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_team_form.py 