# scrape_statcast.py

import pandas as pd
import os
from datetime import datetime, timedelta
from pybaseball import statcast
from pybaseball import cache

cache.disable()

def scrape_statcast_today_or_recent(n_days=3):
    print(f"Attempting to scrape Statcast data for today first, then the last {n_days} days...")

    today = datetime.today()
    successful_scrapes = []

    # Step 1: Try scraping today
    date_str = today.strftime('%Y-%m-%d')
    print(f"Trying today's date: {date_str}")
    try:
        df = statcast(start_dt=date_str, end_dt=date_str)
        if not df.empty:
            output_dir = r"C:\Users\roman\baseball_forecast_project\data\raw"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"statcast_{date_str}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} rows for today to: {output_path}")
            return output_path, date_str
        else:
            print("No data for today, trying previous days...")
    except Exception as e:
        print(f"Error scraping today's data: {e}")

    # Step 2: Scrape recent past days
    for delta in range(1, n_days + 5):
        check_date = today - timedelta(days=delta)
        date_str = check_date.strftime('%Y-%m-%d')
        try:
            print(f"Trying previous date: {date_str}")
            df = statcast(start_dt=date_str, end_dt=date_str)
            if not df.empty:
                output_dir = r"C:\Users\roman\baseball_forecast_project\data\raw"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"statcast_{date_str}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved {len(df)} rows to: {output_path}")
                return output_path, date_str
        except Exception as e:
            print(f"Error scraping data for {date_str}: {e}")

    print("No Statcast data found for today or recent days.")
    return None, None

# Manual test
if __name__ == "__main__":
    scrape_statcast_today_or_recent(n_days=3)

# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrape_statcast.py