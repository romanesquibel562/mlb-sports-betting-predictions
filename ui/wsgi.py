# ui/wsgi.py

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)


# cd C:\Users\roman\baseball_forecast_project\ui
# flask --app wsgi run --debug

#power rankings page:
# http://127.0.0.1:5000/power-rankings

# Gemini Summary test page:
# http://127.0.0.1:5000/api/test_gemini_summary

# Gemini API Key test page:
# http://127.0.0.1:5000/api/test_gemini_key