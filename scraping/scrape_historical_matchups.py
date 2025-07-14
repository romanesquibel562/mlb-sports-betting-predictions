# scrape_historical_matchups.py (rolling-enabled)

from pathlib import Path
import os
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import logging
import argparse

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# <-- NEW: Use pathlib for portability
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "historical_matchups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_output_path(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    return OUTPUT_DIR / f"historical_matchups_{date_str}.csv"

def file_exists_for_date(date_obj):
    return get_output_path(date_obj).exists()

def scrape_historical_matchups(date_obj: datetime) -> pd.DataFrame:
    date_str = date_obj.strftime('%Y-%m-%d')
    url = f"https://www.mlb.com/probable-pitchers/{date_str}"

    if file_exists_for_date(date_obj):
        logger.info(f"Skipping {date_str} — file already exists.")
        return None

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)

    logger.info(f"Scraping matchup data for {date_str} from {url}")

    try:
        driver.get(url)
        driver.implicitly_wait(5)

        # Scroll to bottom to load dynamic content
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
                logger.warning(f"Skipped one matchup due to error: {e}")

        df = pd.DataFrame(matchups)

        if not df.empty:
            output_path = get_output_path(date_obj)
            df.to_csv(output_path, index=False)
            logger.info(f" Saved historical matchups to: {output_path}")
        else:
            logger.warning(f"No matchups found for {date_str}")

        return df

    finally:
        driver.quit()

def scrape_rolling_window(days_back: int):
    end_date = datetime.today() - timedelta(days=1)  # yesterday
    start_date = end_date - timedelta(days=days_back - 1)

    logger.info(f"Scraping rolling window: {start_date.date()} to {end_date.date()}")

    current = start_date
    while current <= end_date:
        try:
            scrape_historical_matchups(current)
        except Exception as e:
            logger.error(f" Failed to scrape {current.strftime('%Y-%m-%d')}: {e}")
        current += timedelta(days=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=int, help="Scrape the past N days ending yesterday (e.g., 90)")
    args = parser.parse_args()

    if args.rolling:
        scrape_rolling_window(args.rolling)
    else:
        # Default: scrape yesterday only
        date = datetime.today() - timedelta(days=1)
        scrape_historical_matchups(date)
        
    # cd C:\Users\roman\baseball_forecast_project\scraping
    # python scrape_historical_matchups.py  --rolling 20
