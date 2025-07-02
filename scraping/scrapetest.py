from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pytz
import time

def run_scrapetest():
    # Setup headless Chrome
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    try:
        # Step 1: Load probable pitchers main page
        driver.get("https://www.mlb.com/probable-pitchers")
        time.sleep(5)  # wait for page to render JS

        # Optional scroll (can help on lazy-loaded sites)
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")
        matchups = []

        # Step 2: Extract all matchup blocks
        all_matchups = soup.select("div.probable-pitchers__matchup")
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern).date()

        for section in all_matchups:
            try:
                # Extract game date
                date_tag = section.select_one("time")
                if not date_tag:
                    continue
                game_date = pd.to_datetime(date_tag["datetime"]).date()
                if game_date != today:
                    continue  # Skip non-today games

                # Extract teams and pitchers
                away_team = section.select_one('.probable-pitchers__team-name--away').text.strip()
                home_team = section.select_one('.probable-pitchers__team-name--home').text.strip()
                pitchers = section.select('.probable-pitchers__pitcher-name-link')
                away_pitcher = pitchers[0].text.strip() if len(pitchers) > 0 else None
                home_pitcher = pitchers[1].text.strip() if len(pitchers) > 1 else None

                matchups.append({
                    "game_date": str(today),
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_pitcher": away_pitcher,
                    "home_pitcher": home_pitcher
                })
            except Exception as e:
                print(f" Skipped a matchup block due to error: {e}")

        # Step 3: Display results
        if not matchups:
            raise ValueError(" No matchups found for today's date.")

        df = pd.DataFrame(matchups)
        print("\n Scraped Matchups Preview:")
        print(df.to_string(index=False))

    finally:
        driver.quit()

if __name__ == "__main__":
    run_scrapetest()

# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrapetest.py