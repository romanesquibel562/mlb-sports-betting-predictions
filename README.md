# MLB Game Outcome Prediction System

## Overview

This repository contains a complete, production-grade machine learning pipeline that forecasts the outcomes of daily MLB games. It leverages real-time Statcast data, historical team and player performance, pitcher trends, and recent team momentum to generate win probability predictions for every scheduled MLB matchup on a given day.

The system is designed to be fully autonomous: it scrapes the latest data, builds features, trains a model using historical outcomes, generates predictions for today's games, and outputs results in a readable CSV format. This project combines web scraping, data engineering, statistical learning, and predictive modeling into a single, deployable solution.

---

## Objective

To develop a scalable and intelligent system capable of predicting daily MLB game outcomes with a high degree of accuracy using advanced statistical features and a robust model architecture. The model is optimized to support:
- Daily sports betting recommendations
- Automated decision-support systems
- Team-level performance forecasts
- Player-centric analysis of influence on win probability

---

## Key Capabilities

- Scrapes and processes real-time MLB matchups and player data
- Collects and merges 30-day rolling Statcast metrics for pitchers and batters
- Calculates team form indicators based on recent win/loss records and run differentials
- Aggregates engineered features into game-level datasets for historical training and daily prediction
- Trains a RandomForestClassifier to classify whether the home team will win
- Outputs a clean, human-readable CSV with win probabilities and betting recommendations

---

## Project Structure and File Descriptions

<pre> ``` 
baseball_forecast_project/
│
├── config/
│   └── ui_weights.yml
│       # Configuration for UI weighting and display logic
│
├── dashboard/
│   # (Reserved for future dashboard or analytics extensions)
│
├── data/
│   # Central data directory (raw + processed outputs)
│
├── data_source/
│   └── simulation.py
│       # Experimental simulation and data source utilities
│
├── evaluation/
│   # Model evaluation utilities and diagnostics
│
├── features/
│   ├── data/
│   │   # Intermediate feature-generation artifacts
│   │
│   ├── build_batter_stat_features.py
│   │   # Builds batter Statcast performance features
│   │
│   ├── build_pitcher_event_features.py
│   │   # Creates event-level pitcher metrics (whiffs, strikeouts, etc.)
│   │
│   ├── build_pitcher_stat_features.py
│   │   # Aggregates recent pitcher Statcast metrics
│   │
│   ├── build_player_event_features.py
│   │   # Event-level Statcast feature engineering
│   │
│   ├── engineer_features.py
│   │   # Feature transformations and aggregations
│   │
│   ├── generate_historical_batter_stats.py
│   │   # Rolling historical batter feature generation
│   │
│   ├── generate_historical_features.py
│   │   # Orchestrates historical feature backfilling
│   │
│   ├── generate_historical_pitcher_stats.py
│   │   # Rolling historical pitcher feature generation
│   │
│   ├── generate_historical_team_form.py
│   │   # Rolling team momentum feature generation
│   │
│   ├── historical_main_features.py
│   │   # Builds unified historical training dataset
│   │
│   └── main_features.py
│       # Builds current-day feature set for predictions
│
├── mlb-sports-betting-predictions/
│   # (Optional exploratory betting analysis outputs)
│
├── modeling/
│   ├── data/
│   │   # Model-ready datasets
│   │
│   ├── evaluate_prediction_accuracy.py
│   │   # Backtesting and accuracy evaluation
│   │
│   ├── generate_power_rankings.py
│   │   # Model-driven team power rankings
│   │
│   ├── predict_today_matchups.py
│   │   # Generates predictions for today’s games
│   │
│   ├── train_model.py
│   │   # Primary ML training pipeline
│   │
│   └── train_xgb.py
│       # XGBoost-based training and calibrated predictions
│
├── plots/
│   # Saved evaluation plots and diagnostics
│
├── scraping/
│   ├── data/
│   │   # Raw scraped outputs
│   │
│   ├── build_odds_file.py
│   │   # Builds sportsbook odds datasets
│   │
│   ├── scrape_game_results.py
│   │   # Scrapes historical MLB game outcomes
│   │
│   ├── scrape_historical_matchups.py
│   │   # Scrapes past matchups for training
│   │
│   ├── scrape_matchups.py
│   │   # Scrapes daily matchups and probable pitchers
│   │
│   ├── scrape_matchups_alt.py
│   │   # Alternate matchup scraper (fallback)
│   │
│   ├── scrape_statcast.py
│   │   # Downloads recent Statcast data
│   │
│   ├── scrape_team_form_mlb.py
│   │   # Scrapes team recent form
│   │
│   ├── scrape_weather.py
│   │   # Scrapes game-day weather data
│   │
│   └── scrapetest.py
│       # Experimental scraping tests
│
├── tests/
│   # Unit and integration tests
│
├── ui/
│   ├── app/
│   │   ├── static/
│   │   │   └── styles.css
│   │   │       # UI styling
│   │   │
│   │   ├── templates/
│   │   │   ├── base.html
│   │   │   ├── index.html
│   │   │   ├── games_index.html
│   │   │   ├── game_detail.html
│   │   │   ├── game_lookup.html
│   │   │   ├── today_ev.html
│   │   │   ├── h2h_lookup.html
│   │   │   ├── h2h_detail.html
│   │   │   └── power_rankings.html
│   │   │
│   │   ├── data_loader.py
│   │   │   # Loads processed model outputs for UI
│   │   │
│   │   ├── gemini_summarizer.py
│   │   │   # LLM-powered game summaries
│   │   │
│   │   ├── routes_api.py
│   │   │   # API endpoints
│   │   │
│   │   └── routes_ui.py
│   │       # Flask UI routes
│   │
│   └── wsgi.py
│       # WSGI entry point
│
├── utils/
│   ├── data/
│   │   # Reference lookup data
│   │
│   ├── build_batter_team_lookup.py
│   │   # Maps batters to teams
│   │
│   └── map_batter_ids.py
│       # Batter ID mapping utilities
│
├── .env
│   # Environment variables (not committed)
│
├── requirements.txt
│   # Python dependencies
│
├── run_daily_pipeline.py
│   # End-to-end orchestration script
│
└── README.md ``` </pre>



---

## Detailed File Descriptions

### `scraping/`

#### `scrape_matchups.py`
Scrapes scheduled MLB matchups (home/away teams and pitchers) from MLB.com for the current day. Outputs a matchup file in the format:
`game_date, home_team, away_team, home_pitcher, away_pitcher`.

#### `scrape_statcast.py`
Downloads Statcast play-by-play data for the last 30 days from the MLB API. This includes every pitch thrown and swing taken, enabling deep performance analysis.

#### `scrape_team_form_mlb.py`
Scrapes recent team performance (wins, losses, run differentials, and win streaks) from public MLB data sources. Outputs `team_form_YYYY-MM-DD.csv` for feature engineering.

#### `scrape_game_results.py`
Scrapes past game outcomes from Baseball Reference to build training labels (`actual_winner`) for historical games. Used in conjunction with past matchups to create the training set.

---

### `features/`

#### `build_pitcher_stat_features.py`
Processes 30-day Statcast data to compute pitcher-level features, including:
- Average velocity
- Spin rate
- Strikeouts
- Whiff rates
- Average bat speed faced

These features are aggregated per pitcher and merged into the game-level dataset.

#### `build_batter_stat_features.py`
Aggregates batter-level Statcast performance by team using a batter-to-team mapping:
- Average launch speed
- Bat speed
- Home runs
- Strikeouts
- Whiff and barrel rates

The final output summarizes overall offensive strength for each team.

#### `map_batter_ids.py`
Matches batter MLBAM IDs from Statcast to their current teams based on inning information and frequency logic. This is necessary for proper team-level aggregation in `build_batter_stat_features.py`.

#### `build_player_event_features.py`
Creates additional player-level metrics from Statcast raw events, including player-specific power, contact, and swing profiles. These are not used directly in modeling but are stored for future extensions.

---

### `modeling/`

#### `historical_main_features.py`
Builds the final historical training dataset (`historical_main_features.csv`) by merging:
- Past matchups (home/away/pitchers)
- Batter stats (by team)
- Pitcher stats
- Team form
- Game outcomes

All features used were collected **only from data available before each game date**, preventing target leakage.

#### `train_model.py`
Trains a RandomForestClassifier on the historical dataset to classify whether the home team wins. Outputs performance metrics:
- Accuracy
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Mean Absolute Percentage Error (MAPE)

Also loads the current day’s features and generates win probabilities and pick recommendations for each scheduled matchup.

---

### `run_daily_pipeline.py`

The core execution script. When run, it:
1. Scrapes today's MLB matchups
2. Scrapes and builds pitcher and batter stats
3. Scrapes team recent form
4. Builds today's `main_features_YYYY-MM-DD.csv`
5. Loads historical dataset and trains the model
6. Predicts win probabilities for today’s matchups
7. Saves readable predictions to CSV

No manual intervention is required. Fully automated.

---

## Model Details

- **Model Type:** Random Forest Classifier
- **Input Features:** 30+
- **Target:** Home team win (binary classification)
- **Evaluation:**
  - Accuracy: 92.4%
  - MAE: 0.076
  - MSE: 0.076
  - MAPE: 7.58%

These results reflect strong predictive power and low error, ideal for sports forecasting use cases.

---

## Example Output

![image](https://github.com/user-attachments/assets/ce48f386-aa72-4eb6-a875-128f87197000)


In the TOR vs NYY matchup, the win probability for Toronto (home team) is 0.34, meaning the model estimates a 66% (1-0.34) chance that the Yankees (away team) will win. Thus, the prediction is "Pick: NYY".

This decision is based on a machine learning model trained on historical data with a backtested accuracy of 92.4%, and a mean absolute error of 0.076.

## Model Accuracy/ Performance
![image](https://github.com/user-attachments/assets/11847757-7d70-4e0e-9ca5-5e274a8fe4f6)

Over multiple game days, this model has achieved an accuracy of ~64%, consistently outperforming naive baselines (e.g., always picking the home team at ~53–55%) and aligning with the upper range of public predictive systems. For reference, industry benchmarks such as ESPN's Elo or FiveThirtyEight’s MLB models typically range between 58–62% accuracy, while Vegas implied probabilities hover around 57–60%. Sustained performance above 60% without relying on betting odds, which indicates strong predictive signal and positions this model near the upper tier of publicly available baseball forecasting tools.

## Outputs
![image](https://github.com/user-attachments/assets/cee0ed7d-87cf-4f6e-9424-9e0cbc70eb9c)
![image](https://github.com/user-attachments/assets/c99563d5-a299-45be-9555-365a5c499aa1)
![image](https://github.com/user-attachments/assets/f3589d1f-768d-4ae7-a3b0-5fa1bb089b5f)
![image](https://github.com/user-attachments/assets/9a884fcf-7d67-43b1-8c2a-69e029c008a8)

## Web UI and Visualization Layer

In addition to the machine learning pipeline, this project includes a fully functional Flask-based web application that enables interactive exploration of model outputs and historical data.

The UI is designed as a read-only analytics dashboard that consumes the same processed CSV outputs generated by the pipeline, ensuring consistency between batch predictions and visual presentation.

### UI Capabilities
- View daily game predictions with win probabilities
- Explore individual game details, including:
  - Starting pitchers
  - Team form
  - Model confidence
- Head-to-head matchup lookup with historical win/loss breakdowns
- Model-driven team power rankings
- AI-generated natural language game summaries using Gemini
- EV-focused views for betting-oriented analysis

### Technical Implementation
- **Framework:** Flask (Blueprint-based architecture)
- **Templates:** Jinja2
- **Styling:** Custom CSS
- **Data Source:** Processed CSV outputs from the modeling pipeline
- **Separation of Concerns:**
  - `routes_ui.py` handles HTML rendering
  - `routes_api.py` exposes structured endpoints
  - `data_loader.py` abstracts CSV ingestion

## Simulation Mode

The system includes a simulation mode designed to support controlled experimentation, scenario analysis, and offline testing without reliance on live MLB data.

Simulation utilities are located in `data_source/simulation.py` and allow the pipeline to be executed against synthetic or historical inputs while preserving the same feature engineering and modeling logic used in production runs.

### Simulation Use Cases
- Testing model behavior under hypothetical scenarios
- Stress-testing feature sensitivity
- Validating pipeline stability without live data dependencies
- Educational demonstrations and reproducible experiments

### Design Philosophy
- Simulation data mirrors the schema of real scraped inputs
- No changes are required to downstream feature engineering or modeling code
- Enables reproducible experimentation and future Monte Carlo extensions

This design allows the system to operate both as a real-time forecasting engine and as a controlled research environment.

## How to Run
1) Clone the repository:

git clone https://github.com/YOUR_USERNAME/mlb-game-prediction.git
cd mlb-game-prediction

2) Set up dependencies:

pip install -r requirements.txt

3) Run the full pipeline:

python run_daily_pipeline.py

4) Run User Interface:

cd ui
python wsgi.py

5) Launch Web Page:

http://127.0.0.1:5000

# User Interface Preview:

<img width="1918" height="811" alt="image" src="https://github.com/user-attachments/assets/4a05641c-a44c-4686-97c4-a0cf0fafb1e9" />


## Use Cases:
Sports betting decision support

Game simulation and forecasting

Player performance tracking

Team strength visualization

Baseball analytics education or portfolio project

## Strengths:
Modular, extensible architecture

No data leakage; true pre-game simulation

Easily generalizable to other sports

Fully automated from scrape to prediction

High predictive performance on real-world data

Interactive Flask-based web UI for model exploration and analysis

Simulation mode for reproducible testing and scenario evaluation


## Future Improvements:
Incorporate weather, park factors, and travel fatigue

Deploy a real-time prediction dashboard (e.g., Streamlit)

Integrate an alert system for best-value picks

Add calibration metrics and ranking confidence intervals

## Author:
Roman Esquibel

Machine Learning Engineer | Sports Analytics Developer | Masters Student at California State University, Fullerton

Email: romanesquib@gmail.com

LinkedIn: https://www.linkedin.com/in/roman-esquibel-75b994223/
