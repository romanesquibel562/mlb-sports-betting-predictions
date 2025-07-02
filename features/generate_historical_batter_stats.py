# generate_historical_batter_stats.py

import os
import glob
import argparse
import pandas as pd
from datetime import datetime
from build_batter_stat_features import build_batter_stat_features

# Paths
statcast_dir = "C:/Users/roman/baseball_forecast_project/data/raw"
output_dir = "C:/Users/roman/baseball_forecast_project/data/processed"
lookup_path = "C:/Users/roman/baseball_forecast_project/utils/data/reference/batter_team_lookup.csv"

def get_existing_dates():
    files = glob.glob(os.path.join(output_dir, "batter_stat_features_*.csv"))
    return {os.path.basename(f).split("_")[-1].replace(".csv", "") for f in files}

def run_rolling_batter_generator(n_days: int):
    existing = get_existing_dates()
    all_statcast_files = sorted(glob.glob(os.path.join(statcast_dir, "statcast_*.csv")))
    processed = 0

    for fpath in all_statcast_files:
        date_str = os.path.basename(fpath).replace("statcast_", "").replace(".csv", "")
        if date_str in existing:
            continue

        print(f"Processing: {date_str}")
        try:
            statcast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            build_batter_stat_features(fpath, lookup_path, statcast_date=statcast_date)  # <-- Fixed
            processed += 1
        except Exception as e:
            print(f"Failed to build batter features for {date_str}: {e}")

        if processed >= n_days:
            break

    print(f"\n Done. Generated batter stats for {processed} new date(s).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=int, default=10, help="Number of new days to process.")
    args = parser.parse_args()

    run_rolling_batter_generator(args.rolling)

    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_batter_stats.py --rolling 20
