# scrape_matchups.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pytz
import time
from pathlib import Path

def run_scrape_matchups():
    # === Paths ===
    BASE_DIR = Path(__file__).resolve().parents[1]
    raw_data_dir = BASE_DIR / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # === Date and URL ===
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).date()
    date_str = today.strftime('%Y-%m-%d')
    url = f"https://www.mlb.com/probable-pitchers/{date_str}"

    # === Web Driver Setup ===
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        driver.implicitly_wait(5)

        # Scroll to bottom to load all content
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")
        matchups = []

        for section in soup.select('div.probable-pitchers__matchup'):
            try:
                away_team = section.select_one('.probable-pitchers__team-name--away').text.strip()
                home_team = section.select_one('.probable-pitchers__team-name--home').text.strip()
                pitchers = section.select('.probable-pitchers__pitcher-name-link')
                away_pitcher = pitchers[0].text.strip() if len(pitchers) > 0 else None
                home_pitcher = pitchers[1].text.strip() if len(pitchers) > 1 else None

                matchups.append({
                    "game_date": date_str,
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_pitcher": away_pitcher,
                    "home_pitcher": home_pitcher
                })
            except Exception as e:
                print(f"Skipped one matchup due to error: {e}")

        df = pd.DataFrame(matchups)
        if df.empty:
            raise ValueError("No matchups were scraped from the site.")

        # Normalize team names
        translation_dict = {
            'Red Sox': 'BOS', 'Yankees': 'NYY', 'Blue Jays': 'TOR', 'Orioles': 'BAL', 'Rays': 'TB',
            'Guardians': 'CLE', 'White Sox': 'CWS', 'Royals': 'KC', 'Tigers': 'DET', 'Twins': 'MIN',
            'Astros': 'HOU', 'Mariners': 'SEA', 'Rangers': 'TEX', 'Angels': 'LAA', 'Athletics': 'OAK',
            'Braves': 'ATL', 'Marlins': 'MIA', 'Mets': 'NYM', 'Phillies': 'PHI', 'Nationals': 'WSH',
            'Brewers': 'MIL', 'Cardinals': 'STL', 'Cubs': 'CHC', 'Pirates': 'PIT', 'Reds': 'CIN',
            'Dodgers': 'LAD', 'Giants': 'SF', 'Padres': 'SD', 'Rockies': 'COL', 'D-backs': 'ARI',
            'Diamondbacks': 'ARI'
        }
        df['home_team'] = df['home_team'].map(translation_dict).fillna(df['home_team'])
        df['away_team'] = df['away_team'].map(translation_dict).fillna(df['away_team'])

        # Double check game_date
        df['game_date'] = pd.to_datetime(df['game_date']).dt.date
        df_today = df[df['game_date'] == today].copy()

        if df_today.empty:
            raise ValueError(f"No matchups found for today's date: {today}")

        # Save output
        output_path = raw_data_dir / f"mlb_probable_pitchers_{today}.csv"
        df_today.to_csv(output_path, index=False)

        return output_path, today

    finally:
        driver.quit()

# === Manual test ===
if __name__ == "__main__":
    path, date = run_scrape_matchups()
    print(f"\nSaved to: {path} | Game date: {date}\n")

    df = pd.read_csv(path)
    print("Scraped Matchups Preview:")
    print(df.to_string(index=False))

    # cd C:\Users\roman\baseball_forecast_project\scraping
    # python scrape_matchups.py
