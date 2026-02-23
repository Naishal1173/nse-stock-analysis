# =========================================================
# DATABASE CONFIG
# =========================================================
# Supports both local development and Render deployment

import os

DB_CONN = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "NseStock"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "root")
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
