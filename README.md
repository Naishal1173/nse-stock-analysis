# NSE Stock Analysis System

A fully automated stock analysis system for NSE (National Stock Exchange) that downloads daily price data, calculates technical indicators, performs backtesting, and provides real-time BUY/SELL signals through an interactive web dashboard.

## 🎯 Key Features

- ✅ **Fully Automated**: Daily data download and indicator calculation via database triggers
- ⚡ **Lightning Fast**: Sub-2ms query performance with optimized data mart architecture
- 📊 **5 Technical Indicators**: SMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator
- 🎯 **Power Signals**: Identifies stocks with multiple agreeing indicators
- 📈 **Backtesting Engine**: Analyzes historical performance with configurable profit targets
- 🔍 **Global Search**: Search any company across the entire database
- 📱 **Responsive Dashboard**: Progressive loading with real-time updates
- 🎨 **Interactive Charts**: TradingView-style candlestick charts with indicators

## 📊 System Status

- **Query Performance**: <2ms average response time
- **Storage Efficiency**: 99.9% reduction (119M → 68K rows)
- **Indicators Tracked**: 23 different indicator configurations
- **System Status**: ✅ FULLY OPERATIONAL

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NaishaH173/nse-stock-analysis.git
   cd nse-stock-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure database**
   ```bash
   # Copy example config
   cp config.example.py config.py
   
   # Edit config.py with your PostgreSQL credentials
   # Update DB_CONN dictionary with your database details
   ```

5. **Setup database schema**
   ```bash
   # Create database tables, functions, and triggers
   python database/setup_optimized_architecture_fast.py
   ```

6. **Download initial data**
   ```bash
   # Downloads latest NSE data and calculates indicators
   python run_daily.py
   ```

7. **Start the web server**
   ```bash
   python run_api.py
   ```

8. **Open dashboard**
   ```
   Navigate to: http://localhost:8000
   ```

### Daily Usage

Run this command daily (or set up a cron job/scheduled task):
```bash
python run_daily.py
```

This automatically:
1. Downloads new NSE price data (Bhavcopy)
2. Inserts into database
3. Triggers automatic indicator calculation
4. Refreshes latest BUY signals table
5. Exports CSV report

## 🏗️ System Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  AUTOMATIC DAILY WORKFLOW                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. python run_daily.py                                     │
│     ↓                                                       │
│  2. Downloads Bhavcopy from NSE                             │
│     ↓                                                       │
│  3. Inserts into daily_prices table                         │
│     ↓                                                       │
│  4. DATABASE TRIGGER FIRES (trg_auto_update_indicators)     │
│     ↓                                                       │
│  5. Calls usp_master_update_all_indicators()                │
│     ↓                                                       │
│  6. Calculates all indicators incrementally:                │
│     • SMA (5, 10, 20, 30, 50, 100, 200)                     │
│     • RSI (7, 14, 21, 50, 80)                               │
│     • MACD (Short, Long, Standard)                          │
│     • Bollinger Bands (10, 20, 50, 100)                     │
│     • Stochastic (5, 9, 14, 21, 50)                         │
│     ↓                                                       │
│  7. Calls refresh_latest_buy_signals()                      │
│     ↓                                                       │
│  8. Exports CSV report to exports/                          │
│     ↓                                                       │
│  9. ✅ Dashboard ready with fresh signals!                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with custom triggers and stored procedures
- **Frontend**: Vanilla JavaScript with Jinja2 templates
- **Charts**: Lightweight Charts library (TradingView-style)
- **Data Source**: NSE India (National Stock Exchange)

## 📁 Project Structure

```
nse-stock-analysis/
├── app/
│   └── api.py                          # FastAPI application with all endpoints
├── pipeline/
│   └── bhavcopy_downloader.py          # NSE data downloader
├── static/
│   ├── dashboard-new.css               # Dashboard styles
│   ├── script.js                       # Symbol page logic
│   ├── progressive-fast.js             # Dashboard progressive loading
│   ├── style.css                       # Symbol page styles
│   └── lightweight-charts.js           # Chart library
├── templates/
│   ├── base.html                       # Base template
│   ├── index.html                      # Dashboard page
│   ├── symbol.html                     # Individual stock analysis page
│   └── diagnostic.html                 # System diagnostics
├── logs/                               # Automation logs (auto-generated)
├── bhavdata/                           # Downloaded NSE data (auto-generated)
├── exports/                            # CSV exports (auto-generated)
├── config.example.py                   # Configuration template
├── config.py                           # Your configuration (not in git)
├── requirements.txt                    # Python dependencies
├── run_daily.py                        # Daily automation script
├── run_api.py                          # Start web server
├── export_all_signals.py               # Export signals to CSV
└── README.md                           # This file
```

## 🗄️ Database Schema

### Core Tables

- **symbols** - Stock symbols and metadata
- **daily_prices** - OHLC (Open, High, Low, Close) price data
- **smatbl** - Simple Moving Average indicators
- **rsitbl** - Relative Strength Index indicators
- **macdtbl** - MACD indicators
- **bbtbl** - Bollinger Bands indicators
- **stochtbl** - Stochastic Oscillator indicators

### Optimized Tables (High Performance)

- **latest_buy_signals** - Only latest date BUY signals (~2K rows, <2ms queries)
- **mv_all_signals** - Materialized view of all latest signals (~68K rows)

### Key Database Functions

- `refresh_latest_buy_signals()` - Refreshes dashboard table with latest BUY signals
- `refresh_all_signal_views()` - Refreshes all materialized views
- `usp_master_update_all_indicators()` - Master procedure that calculates all indicators
- `usp_incremental_update_sma()` - Incremental SMA calculation
- `usp_incremental_update_rsi()` - Incremental RSI calculation
- `usp_incremental_update_macd()` - Incremental MACD calculation
- `usp_incremental_update_bb()` - Incremental Bollinger Bands calculation
- `usp_incremental_update_stoch()` - Incremental Stochastic calculation

### Database Trigger

- **trg_auto_update_indicators** - Automatically fires when new data inserted into `daily_prices`
  - Calls `usp_master_update_all_indicators()`
  - Ensures indicators are always up-to-date
  - No manual intervention required

## 🔌 API Endpoints

### Dashboard Endpoints

- `GET /` - Main dashboard with latest BUY signals
- `GET /symbol/{symbol}` - Individual stock analysis page
- `GET /diagnostic` - System diagnostics and health check

### Data Endpoints

- `GET /api/symbols?q={query}` - Search companies (global search)
- `GET /api/signal-summary` - Summary statistics of all signals
- `GET /api/signals-by-indicator` - Signal counts grouped by indicator
- `GET /api/latest-signals?indicator={name}` - Latest BUY signals (filtered)
- `GET /api/symbol/{symbol}/indicators` - All indicators for a specific stock
- `GET /api/symbol/{symbol}/chart?days={n}` - Price and indicator data for charts
- `GET /api/indicators` - List of all available indicators (dynamic from DB)
- `GET /api/power-signals?min_signals={n}` - Stocks with multiple agreeing indicators
- `GET /api/signal-statistics` - Detailed statistics by indicator type
- `GET /api/metrics` - System performance metrics

### Analysis Endpoints (Backtesting)

- `GET /api/analyze-all?target={%}&days={n}&limit={n}` - Analyze all current BUY signals
- `GET /api/analyze-indicator?indicator={name}&target={%}&days={n}` - Analyze specific indicator
- `GET /api/analyze-by-type?indicator_type={type}&target={%}&days={n}` - Analyze by type (SMA, RSI, etc.)
- `GET /api/analyze-power-signals?min_signals={n}&target={%}&days={n}` - Analyze power signals
- `GET /api/analyze-progressive?target={%}&days={n}` - Progressive analysis with streaming
- `GET /api/analyze-fast?target={%}&days={n}` - Fast parallel analysis

### Health Check

- `GET /api/health` - System health status

### Query Parameters

- `target` - Profit target percentage (default: 5.0)
- `days` - Analysis window in days (default: 30)
- `limit` - Maximum results to return
- `min_signals` - Minimum number of agreeing indicators (default: 3)
- `indicator` - Specific indicator name (e.g., "SMA10", "RSI14")
- `indicator_type` - Indicator category (SMA, RSI, BB, MACD, STOCH)
- `q` - Search query string

## 📊 Technical Indicators

### Simple Moving Average (SMA)
- Periods: 5, 10, 20, 30, 50, 100, 200 days
- BUY Signal: Price crosses above SMA
- SELL Signal: Price crosses below SMA

### Relative Strength Index (RSI)
- Periods: 7, 14, 21, 50, 80 days
- BUY Signal: RSI crosses above 30 (oversold)
- SELL Signal: RSI crosses above 70 (overbought)

### MACD (Moving Average Convergence Divergence)
- Types: Short, Long, Standard
- BUY Signal: MACD line crosses above signal line
- SELL Signal: MACD line crosses below signal line

### Bollinger Bands
- Periods: 10, 20, 50, 100 days
- BUY Signal: Price touches or crosses below lower band
- SELL Signal: Price touches or crosses above upper band

### Stochastic Oscillator
- Periods: 5, 9, 14, 21, 50 days
- BUY Signal: %K crosses above %D in oversold region (<20)
- SELL Signal: %K crosses below %D in overbought region (>80)

## 🎯 Backtesting Engine

The system includes a sophisticated backtesting engine that analyzes historical performance of BUY signals:

### Features

- **Configurable Profit Target**: Set your desired profit percentage (default: 5%)
- **Time Window**: Analyze performance over N days (default: 30 days)
- **Trade Status Tracking**:
  - **SUCCESS**: Target profit achieved within time window
  - **FAIL**: Time window completed without hitting target
  - **OPEN**: Insufficient data (signal too recent)

### Metrics Calculated

- **Success Rate**: (Successful Trades / Completed Trades) × 100
- **Avg Max Profit**: Average maximum profit from winning trades
- **Avg Max Loss**: Average maximum loss from losing trades
- **Total Signals**: Total number of BUY signals analyzed
- **Completed Trades**: Signals with full data window
- **Open Trades**: Recent signals still being tracked

### Example Analysis

```
Indicator: SMA10
Target: 5.0% profit in 30 days

Results:
- Total Signals: 200
- Completed Trades: 185
- Open Trades: 15
- Successful: 120
- Failed: 65
- Success Rate: 64.86%
- Avg Max Profit: 8.45%
- Avg Max Loss: -3.21%
```

## 🎨 Dashboard Features

### Main Dashboard
- **Progressive Loading**: Loads data in batches for instant initial display
- **Real-time Search**: Global company search with autocomplete
- **Signal Filtering**: Filter by indicator type
- **Power Signals**: Highlight stocks with multiple agreeing indicators
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Auto-refresh**: Updates automatically when new data available

### Symbol Analysis Page
- **Interactive Charts**: TradingView-style candlestick charts
- **Indicator Overlays**: Visualize SMA, RSI, MACD, BB, Stochastic
- **Backtesting Results**: Historical performance analysis
- **Trade Status**: SUCCESS/FAIL/OPEN status for each signal
- **Detailed Metrics**: Success rate, avg profit/loss, trade counts
- **Dynamic Indicators**: Only shows indicators with actual BUY signals

### Performance Optimizations
- **Data Mart Architecture**: Query only latest BUY signals (~2K rows)
- **Database Indexing**: Optimized indexes for sub-2ms queries
- **Caching**: 5-minute cache for indicator lists
- **Batch Loading**: Load prices in bulk for analysis
- **Parallel Processing**: Multi-threaded backtesting analysis

## � Configuration

### Database Configuration

Edit `config.py` (copy from `config.example.py`):

```python
DB_CONN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_database_name",
    "user": "your_username",
    "password": "your_password"
}
```

### Indicator Settings

Customize indicator periods in `config.py`:

```python
# SMA periods
SMA_PERIODS = [5, 10, 20, 30, 50, 100, 200]

# RSI periods and thresholds
RSI_PERIODS = [7, 14, 21, 50, 80]
RSI_BUY_LEVEL = 30
RSI_SELL_LEVEL = 70

# Historical data lookback
MAX_LOOKBACK_DAYS = 400
```

### NSE Data Fetching

Configure NSE API settings:

```python
NSE_BASE_URL = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
NSE_HOME_URL = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}

# Fetch settings
RECENT_DAYS = 15
FALLBACK_DAYS = 10
DELAY_MIN = 2.5
DELAY_MAX = 4.5
```

## 🛠️ Troubleshooting

### Dashboard Shows No Data

```bash
# Refresh the latest signals table
python -c "import psycopg2; from config import DB_CONN; conn = psycopg2.connect(**DB_CONN); cur = conn.cursor(); cur.execute('SELECT refresh_latest_buy_signals()'); conn.commit(); print('Refreshed!')"
```

### Data Not Updating

```bash
# Run daily automation manually
python run_daily.py

# Check logs
cat logs/daily_automation_*.log | tail -50
```

### Database Trigger Not Working

```sql
-- Check trigger status
SELECT tgname, tgenabled 
FROM pg_trigger 
WHERE tgname = 'trg_auto_update_indicators';

-- Re-enable if disabled
ALTER TABLE daily_prices ENABLE TRIGGER trg_auto_update_indicators;
```

### Slow Query Performance

```sql
-- Verify indexes exist
SELECT tablename, indexname 
FROM pg_indexes 
WHERE tablename = 'latest_buy_signals';

-- Refresh materialized view
SELECT refresh_all_signal_views();
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

### Port Already in Use

```bash
# Change port in run_api.py
# Find line: uvicorn.run(app, host="0.0.0.0", port=8000)
# Change to: uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 📝 Maintenance

### Manual Refresh Commands

```bash
# Refresh latest BUY signals table
python -c "import psycopg2; from config import DB_CONN; conn = psycopg2.connect(**DB_CONN); cur = conn.cursor(); cur.execute('SELECT refresh_latest_buy_signals()'); conn.commit(); print('Done!')"

# Export all signals to CSV
python export_all_signals.py

# Run full daily automation
python run_daily.py
```

### Database Maintenance

```sql
-- Vacuum and analyze tables
VACUUM ANALYZE daily_prices;
VACUUM ANALYZE latest_buy_signals;

-- Refresh materialized views
SELECT refresh_all_signal_views();

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Backup Database

```bash
# Backup entire database
pg_dump -U postgres -d your_database_name > backup_$(date +%Y%m%d).sql

# Backup specific tables
pg_dump -U postgres -d your_database_name -t daily_prices -t symbols > backup_core_$(date +%Y%m%d).sql

# Restore from backup
psql -U postgres -d your_database_name < backup_20260220.sql
```

### Scheduled Automation (Optional)

#### Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 6:30 PM
4. Action: Start a program
5. Program: `C:\path\to\venv\Scripts\python.exe`
6. Arguments: `C:\path\to\run_daily.py`
7. Start in: `C:\path\to\project`

#### Linux Cron Job
```bash
# Edit crontab
crontab -e

# Add line (runs daily at 6:30 PM)
30 18 * * * cd /path/to/project && /path/to/venv/bin/python run_daily.py >> logs/cron.log 2>&1
```

## 📈 Performance Metrics

### Before Optimization
- Query time: **5-30 seconds**
- Rows scanned: **119,872,791**
- Storage: **Full historical data**
- Dashboard: **Slow and frustrating**

### After Optimization
- Query time: **<2ms**
- Rows scanned: **~2,000**
- Storage: **Data mart with latest signals only**
- Dashboard: **Instant response!**

**Performance Improvement: 1000x faster! 🚀**

### Architecture Benefits
- **Data Mart Pattern**: Separate optimized table for dashboard queries
- **Incremental Updates**: Only calculate new data, not full history
- **Database Triggers**: Automatic updates without manual intervention
- **Materialized Views**: Pre-computed aggregations
- **Smart Indexing**: Optimized indexes on frequently queried columns
- **Batch Processing**: Bulk operations for backtesting

## 🎓 How It Works

### Automatic Trigger System

When new price data is inserted into `daily_prices`:

1. **Trigger fires**: `trg_auto_update_indicators` executes automatically
2. **Master procedure**: Calls `usp_master_update_all_indicators()`
3. **Incremental updates**: Each indicator procedure runs:
   - `usp_incremental_update_sma()` - Only calculates new SMA values
   - `usp_incremental_update_rsi()` - Only calculates new RSI values
   - `usp_incremental_update_macd()` - Only calculates new MACD values
   - `usp_incremental_update_bb()` - Only calculates new BB values
   - `usp_incremental_update_stoch()` - Only calculates new Stochastic values
4. **Refresh signals**: Calls `refresh_latest_buy_signals()`
5. **Dashboard ready**: Latest BUY signals available instantly

### Data Mart Approach

Instead of querying 119M rows across 5 indicator tables, we maintain a small "Data Mart" table (`latest_buy_signals`) with only:
- Latest trading date signals
- BUY signals only (no SELL or NULL)
- ~2,000 rows total
- Optimized indexes for instant queries
- Automatically refreshed by database trigger

This architectural pattern provides:
- **Instant queries**: <2ms response time
- **Always current**: Updates automatically with new data
- **Simple queries**: No complex joins or aggregations needed
- **Scalable**: Performance doesn't degrade with historical data growth

## 🔐 Security & Best Practices

### Configuration Security
- Never commit `config.py` with real credentials (already in `.gitignore`)
- Use strong database passwords
- Restrict database access to necessary IPs only
- Consider using environment variables for production

### Database Security
```python
# Use environment variables (optional)
import os

DB_CONN = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "nse_data"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD")
}
```

### API Security (Production)
- Add authentication/authorization if exposing publicly
- Use HTTPS in production
- Implement rate limiting
- Add CORS restrictions
- Monitor for unusual activity

### Data Privacy
- NSE data is publicly available
- No personal user data is stored
- Logs contain only system information
- Exports contain only stock data

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**: Ensure all functionality works
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow existing code style
- Add comments for complex logic
- Test with real data before submitting
- Update README if adding new features
- Keep commits focused and atomic

## 📄 License

This project is for educational and personal use. Please ensure compliance with NSE data usage terms and conditions.

### Disclaimer
- This system is for educational purposes only
- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research before investing
- The authors are not responsible for any trading losses

## 🙏 Acknowledgments

- **NSE India** - For providing public market data
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Powerful open-source database
- **Lightweight Charts** - TradingView-style charting library
- **Python Community** - For excellent libraries and tools

## 📞 Support & Contact

### Issues
If you encounter any problems:
1. Check the Troubleshooting section
2. Review logs in `logs/` directory
3. Verify database trigger status
4. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information
   - Relevant log excerpts

### Questions
For questions or discussions:
- Open a GitHub Discussion
- Check existing issues for similar problems
- Review the documentation thoroughly

## 🗺️ Roadmap

Future enhancements planned:
- [ ] Email/SMS alerts for power signals
- [ ] Mobile app (React Native)
- [ ] More technical indicators (Fibonacci, Ichimoku, etc.)
- [ ] Portfolio tracking
- [ ] Backtesting strategy builder
- [ ] Machine learning predictions
- [ ] Multi-exchange support (BSE, etc.)
- [ ] Real-time WebSocket updates
- [ ] User authentication and personalization
- [ ] Advanced charting tools

## 📊 Project Stats

- **Lines of Code**: ~3,500+
- **Database Tables**: 8 core + 2 optimized
- **API Endpoints**: 20+
- **Technical Indicators**: 23 configurations
- **Performance**: <2ms query time
- **Storage Efficiency**: 99.9% reduction

---

**System Status: ✅ FULLY OPERATIONAL**

Made with ❤️ for the trading community

Last Updated: February 2026
