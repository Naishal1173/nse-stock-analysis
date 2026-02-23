# =========================================================
# DATABASE CONFIG
# =========================================================
# Copy this file to config.py and update with your settings

DB_CONN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_database_name",
    "user": "your_username",
    "password": "your_password"
}

print("CONNECTED")

# =========================================================
# NSE CONFIG
# =========================================================

NSE_BASE_URL = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
NSE_HOME_URL = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}


# =========================================================
# INDICATOR SETTINGS
# =========================================================

SMA_PERIODS = [5, 10, 20, 30, 50, 100, 200]
RSI_PERIODS = [7, 14, 21, 50, 80]

RSI_BUY_LEVEL = 30
RSI_SELL_LEVEL = 70

MAX_LOOKBACK_DAYS = 400


# =========================================================
# FETCH SETTINGS
# =========================================================

RECENT_DAYS = 15
FALLBACK_DAYS = 10

DELAY_MIN = 2.5
DELAY_MAX = 4.5

SESSION_REFRESH_EVERY = 30
