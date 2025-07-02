# scrape_game_results.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Normalization to match historical_matchups (non-abbreviated names)
full_name_to_short = {
    "Arizona Diamondbacks": "D-backs",
    "Atlanta Braves": "Braves",
    "Baltimore Orioles": "Orioles",
    "Boston Red Sox": "Red Sox",
    "Chicago White Sox": "White Sox",
    "Chicago Cubs": "Cubs",
    "Cincinnati Reds": "Reds",
    "Cleveland Guardians": "Guardians",
    "Colorado Rockies": "Rockies",
    "Detroit Tigers": "Tigers",
    "Houston Astros": "Astros",
    "Kansas City Royals": "Royals",
    "Los Angeles Angels": "Angels",
    "Los Angeles Dodgers": "Dodgers",
    "Miami Marlins": "Marlins",
    "Milwaukee Brewers": "Brewers",
    "Minnesota Twins": "Twins",
    "New York Mets": "Mets",
    "New York Yankees": "Yankees",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Phillies",
    "Pittsburgh Pirates": "Pirates",
    "San Diego Padres": "Padres",
    "San Francisco Giants": "Giants",
    "Seattle Mariners": "Mariners",
    "St. Louis Cardinals": "Cardinals",
    "Tampa Bay Rays": "Rays",
    "Texas Rangers": "Rangers",
    "Toronto Blue Jays": "Blue Jays",
    "Washington Nationals": "Nationals"
}

def normalize(name):
    return full_name_to_short.get(name.strip(), None)

def scrape_results_for_date(game_date, matchup_csv_path, output_dir):
    try:
        result_path = os.path.join(output_dir, f"historical_results_{game_date}.csv")
        if os.path.exists(result_path):
            logger.info(f"Skipping {game_date} — results already exist.")
            return

        matchups = pd.read_csv(matchup_csv_path)
        matchups["game_date"] = game_date
        logger.info(f"Loaded {len(matchups)} matchups from {matchup_csv_path}")

        year, month, day = game_date.year, game_date.month, game_date.day
        url = f"https://www.baseball-reference.com/boxes/index.fcgi?year={year}&month={month}&day={day}"
        logger.info(f"Loading URL: {url}")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        results = []

        for game in soup.select("div.game_summary"):
            try:
                winner_tag = game.select_one("tr.winner td a")
                loser_tag = game.select_one("tr.loser td a")
                if not winner_tag or not loser_tag:
                    continue

                winner = normalize(winner_tag.text)
                loser = normalize(loser_tag.text)

                if not winner or not loser:
                    logger.warning(f"Could not normalize team names: {winner_tag.text} vs {loser_tag.text}")
                    continue

                for _, row in matchups.iterrows():
                    if {winner, loser} == {row["home_team"], row["away_team"]}:
                        results.append({
                            "game_date": game_date,
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "winner": winner
                        })
                        break

            except Exception as e:
                logger.warning(f"Error parsing game: {e}")

        if not results:
            logger.warning(f"No results matched the matchups for {game_date}")
            return

        results_df = pd.DataFrame(results)
        results_df.to_csv(result_path, index=False)
        logger.info(f"Saved {len(results_df)} results to: {result_path}")

    except Exception as e:
        logger.error(f"Error processing {game_date}: {e}")

def run_rolling_scraper():
    today = datetime.today().date()
    raw_dir = "C:/Users/roman/baseball_forecast_project/data/raw/historical_matchups"
    output_dir = "C:/Users/roman/baseball_forecast_project/data/processed"

    for delta in range(1, 21):  # Last 20 days (excluding today)
        date = today - timedelta(days=delta)
        matchup_file = os.path.join(raw_dir, f"historical_matchups_{date}.csv")
        if os.path.exists(matchup_file):
            scrape_results_for_date(date, matchup_file, output_dir)
        else:
            logger.info(f"No matchup file found for {date}, skipping.")

# Entry point
if __name__ == "__main__":
    run_rolling_scraper()


# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrape_game_results.py --rolling 20