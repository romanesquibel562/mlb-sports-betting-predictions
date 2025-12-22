# ui/wsgi.py

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)


# =============================================================================
# cd C:\Users\roman\baseball_forecast_project\ui
# flask --app wsgi run --debug

# Home page:
# http://127.0.0.1:5000/

# Game summaries page:
# http://127.0.0.1:5000/game-summaries

# http://127.0.0.1:5000/game-lookup

# http://127.0.0.1:5000/head-to-head
# =============================================================================