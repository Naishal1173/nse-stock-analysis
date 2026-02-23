#!/usr/bin/env python3
"""
NSE Stock Data Daily Automation
Downloads price data, calculates indicators, generates signals, and verifies results
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path
import psycopg2
from config import DB_CONN
import pandas as pd
import os

# ======================================================
# CONFIGURATION
# ======================================================
PIPELINE_DIR = Path("pipeline")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

# ======================================================
# DATABASE FUNCTIONS
# ======================================================
def get_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=DB_CONN["host"],
        port=DB_CONN["port"],
        dbname=DB_CONN["dbname"],
        user=DB_CONN["user"],
        password=DB_CONN["password"]
    )

def log_message(message, log_file=None):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    try:
        print(log_msg)
    except UnicodeEncodeError:
        print(log_msg.encode('ascii', 'ignore').decode('ascii'))
    
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

def run_script(script_path, args=None, log_file=None):
    """Run a Python script and capture output"""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    log_message(f"Running: {' '.join(cmd)}", log_file)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            log_message(f"Output:\n{result.stdout}", log_file)
        
        if result.stderr:
            log_message(f"Errors:\n{result.stderr}", log_file)
        
        return result.returncode == 0
            
    except Exception as e:
        log_message(f"Error running script: {e}", log_file)
        return False

def check_database_status():
    """Check database connectivity and stats"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(trade_date) FROM daily_prices")
        latest_price_date = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE trade_date = %s", (latest_price_date,))
        price_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(trade_date) FROM smatbl")
        latest_indicator_date = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM smatbl WHERE trade_date = %s", (latest_indicator_date,))
        indicator_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM symbols")
        symbols_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            "status": "connected",
            "latest_price_date": latest_price_date,
            "price_count": price_count,
            "latest_indicator_date": latest_indicator_date,
            "indicator_count": indicator_count,
            "symbols_count": symbols_count
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def verify_results(trade_date, log_file=None):
    """Verify calculation results - Check all indicator tables"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        log_message(f"\nVerifying results for {trade_date}...", log_file)
        
        # Check SMA table
        cursor.execute("""
            SELECT COUNT(*) FROM smatbl
            WHERE trade_date = %s
        """, (trade_date,))
        sma_count = cursor.fetchone()[0]
        
        # Check RSI table
        cursor.execute("""
            SELECT COUNT(*) FROM rsitbl
            WHERE trade_date = %s
        """, (trade_date,))
        rsi_count = cursor.fetchone()[0]
        
        # Check Bollinger Bands table
        cursor.execute("""
            SELECT COUNT(*) FROM bbtbl
            WHERE trade_date = %s
        """, (trade_date,))
        bb_count = cursor.fetchone()[0]
        
        # Check MACD table
        cursor.execute("""
            SELECT COUNT(*) FROM macdtbl
            WHERE trade_date = %s
        """, (trade_date,))
        macd_count = cursor.fetchone()[0]
        
        # Check Stochastic table
        cursor.execute("""
            SELECT COUNT(*) FROM stochtbl
            WHERE trade_date = %s
        """, (trade_date,))
        stoch_count = cursor.fetchone()[0]
        
        log_message(f"\nIndicator records for {trade_date}:", log_file)
        log_message(f"  SMA records: {sma_count:,}", log_file)
        log_message(f"  RSI records: {rsi_count:,}", log_file)
        log_message(f"  Bollinger Bands records: {bb_count:,}", log_file)
        log_message(f"  MACD records: {macd_count:,}", log_file)
        log_message(f"  Stochastic records: {stoch_count:,}", log_file)
        
        # Check symbols with complete data
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol_id) FROM daily_prices
            WHERE trade_date = %s
        """, (trade_date,))
        total_symbols = cursor.fetchone()[0]
        
        log_message(f"\nTotal symbols with price data: {total_symbols:,}", log_file)
        
        # Verify all tables have data
        all_tables_have_data = sma_count > 0 and rsi_count > 0 and bb_count > 0 and macd_count > 0 and stoch_count > 0
        
        if not all_tables_have_data:
            log_message("ERROR: One or more indicator tables are empty", log_file)
            cursor.close()
            conn.close()
            return False
        
        # Check signals in tables
        cursor.execute("""
            SELECT signal, COUNT(*) as count
            FROM smatbl
            WHERE trade_date = %s
            GROUP BY signal
            ORDER BY signal
        """, (trade_date,))
        
        signals = cursor.fetchall()
        log_message(f"\nSignals in SMA table:", log_file)
        has_signals = False
        for signal, count in signals:
            signal_type = signal if signal else 'NULL'
            log_message(f"  {signal_type}: {count:,}", log_file)
            if signal in ['BUY', 'SELL']:
                has_signals = True
        
        cursor.close()
        conn.close()
        
        if not has_signals:
            log_message("WARNING: No BUY/SELL signals found", log_file)
        
        # Success if all tables have data
        success = all_tables_have_data
        
        if success:
            log_message("\n✅ Verification: SUCCESS - All indicator tables populated", log_file)
        else:
            log_message("\n❌ Verification: FAILED - Some indicator tables are empty", log_file)
        
        return success
        
    except Exception as e:
        log_message(f"❌ Verification error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return False

def export_all_buy_signals(log_file=None, latest_date=None):
    """Export or append BUY signals to CSV - Incremental updates"""
    try:
        log_message("\n" + "="*70, log_file)
        log_message("UPDATING BUY SIGNALS CSV", log_file)
        log_message("="*70, log_file)
        
        conn = get_connection()
        
        # Find existing CSV file
        csv_file = EXPORT_DIR / "All_BUY_Signals_Complete.csv"
        
        if csv_file.exists():
            # File exists - append only new date's data
            log_message(f"\n📁 Existing file found: {csv_file.name}", log_file)
            
            # Read existing CSV to find last date
            existing_df = pd.read_csv(csv_file)
            last_date_in_csv = existing_df['trade_date'].max()
            log_message(f"📅 Last date in CSV: {last_date_in_csv}", log_file)
            
            if latest_date and latest_date > last_date_in_csv:
                # Query only NEW data for the latest date
                log_message(f"📊 Fetching NEW data for: {latest_date}", log_file)
                
                query = """
                SELECT 
                    s.symbol,
                    combined.trade_date,
                    combined.indicator,
                    combined.value,
                    'BUY' as signal
                FROM (
                    SELECT symbol_id, trade_date, indicator, value FROM smatbl WHERE signal = 'BUY' AND trade_date = %s
                    UNION ALL
                    SELECT symbol_id, trade_date, indicator, value FROM rsitbl WHERE signal = 'BUY' AND trade_date = %s
                    UNION ALL
                    SELECT symbol_id, trade_date, indicator, value FROM bbtbl WHERE signal = 'BUY' AND trade_date = %s
                    UNION ALL
                    SELECT symbol_id, trade_date, indicator_set as indicator, macd_line as value FROM macdtbl WHERE signal = 'BUY' AND trade_date = %s
                    UNION ALL
                    SELECT symbol_id, trade_date, indicator, k_value as value FROM stochtbl WHERE signal = 'BUY' AND trade_date = %s
                ) combined
                JOIN symbols s ON combined.symbol_id = s.symbol_id
                ORDER BY s.symbol, combined.trade_date DESC, combined.indicator
                """
                
                new_df = pd.read_sql_query(query, conn, params=(latest_date, latest_date, latest_date, latest_date, latest_date))
                
                if len(new_df) > 0:
                    log_message(f"✓ Retrieved {len(new_df):,} NEW BUY signals", log_file)
                    
                    # Append to existing CSV
                    log_message(f"💾 Appending to existing CSV...", log_file)
                    new_df.to_csv(csv_file, mode='a', header=False, index=False)
                    
                    # Get updated file info
                    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
                    total_records = len(existing_df) + len(new_df)
                    
                    log_message(f"\n✅ Append Complete!", log_file)
                    log_message(f"📊 File: {csv_file.name}", log_file)
                    log_message(f"💾 Size: {file_size:.2f} MB", log_file)
                    log_message(f"📈 NEW BUY Signals Added: {len(new_df):,}", log_file)
                    log_message(f"📈 Total BUY Signals: {total_records:,}", log_file)
                    log_message(f"📅 Date Range: 2016-01-07 to {latest_date}", log_file)
                else:
                    log_message(f"ℹ️  No BUY signals found for {latest_date}", log_file)
            else:
                log_message(f"ℹ️  CSV is already up-to-date (latest date: {last_date_in_csv})", log_file)
        
        else:
            # File doesn't exist - create complete export
            log_message(f"\n📁 Creating new file: {csv_file.name}", log_file)
            log_message("⏳ Exporting all historical BUY signals (2016-2026)...", log_file)
            
            # Query to get ALL BUY signals - SORTED BY COMPANY FIRST
            query = """
            SELECT 
                s.symbol,
                combined.trade_date,
                combined.indicator,
                combined.value,
                'BUY' as signal
            FROM (
                SELECT symbol_id, trade_date, indicator, value FROM smatbl WHERE signal = 'BUY'
                UNION ALL
                SELECT symbol_id, trade_date, indicator, value FROM rsitbl WHERE signal = 'BUY'
                UNION ALL
                SELECT symbol_id, trade_date, indicator, value FROM bbtbl WHERE signal = 'BUY'
                UNION ALL
                SELECT symbol_id, trade_date, indicator_set as indicator, macd_line as value FROM macdtbl WHERE signal = 'BUY'
                UNION ALL
                SELECT symbol_id, trade_date, indicator, k_value as value FROM stochtbl WHERE signal = 'BUY'
            ) combined
            JOIN symbols s ON combined.symbol_id = s.symbol_id
            ORDER BY s.symbol, combined.trade_date DESC, combined.indicator
            """
            
            log_message("📊 Fetching all data from database...", log_file)
            df = pd.read_sql_query(query, conn)
            log_message(f"✓ Retrieved {len(df):,} BUY signals", log_file)
            
            log_message(f"💾 Writing to CSV file...", log_file)
            df.to_csv(csv_file, index=False)
            
            # Get file info
            file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
            min_date = df['trade_date'].min()
            max_date = df['trade_date'].max()
            unique_companies = df['symbol'].nunique()
            unique_indicators = df['indicator'].nunique()
            
            log_message(f"\n✅ Export Complete!", log_file)
            log_message(f"📊 File: {csv_file.name}", log_file)
            log_message(f"💾 Size: {file_size:.2f} MB", log_file)
            log_message(f"📈 Total BUY Signals: {len(df):,}", log_file)
            log_message(f"📅 Date Range: {min_date} to {max_date}", log_file)
            log_message(f"🏢 Companies: {unique_companies:,}", log_file)
            log_message(f"📊 Indicators: {unique_indicators}", log_file)
            log_message(f"✨ Data grouped by company (all dates per company)", log_file)
        
        conn.close()
        return True
        
    except Exception as e:
        log_message(f"❌ Export error: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return False

# ======================================================
# MAIN AUTOMATION
# ======================================================
def run_daily_automation():
    """Run complete daily automation"""
    start_time = datetime.now()
    log_file = LOG_DIR / f"daily_automation_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
    
    log_message("="*70, log_file)
    log_message("NSE DAILY AUTOMATION STARTED", log_file)
    log_message("="*70, log_file)
    log_message(f"Start Time: {start_time}", log_file)
    
    # Step 1: Check database
    log_message("\nStep 1: Checking database status...", log_file)
    status = check_database_status()
    
    if status["status"] == "error":
        log_message(f"ERROR: Database check failed: {status['error']}", log_file)
        return False
    
    log_message(f"Latest price date: {status['latest_price_date']}", log_file)
    log_message(f"Latest indicator date: {status['latest_indicator_date']}", log_file)
    log_message(f"Total symbols: {status['symbols_count']:,}", log_file)
    
    # Step 2: Download price data
    log_message("\nStep 2: Downloading price data (Bhavcopy)...", log_file)
    price_success = run_script(PIPELINE_DIR / "bhavcopy_downloader.py", log_file=log_file)
    
    if not price_success:
        log_message("ERROR: Price data download failed", log_file)
        return False
    
    log_message("Price data download completed", log_file)
    
    # Step 2.5: Verify new trading data exists (CRITICAL)
    log_message("\nStep 2.5: Verifying new trading data...", log_file)
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(trade_date) FROM daily_prices")
    latest_price_date = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(trade_date) FROM smatbl")
    latest_indicator_date = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    if latest_price_date <= latest_indicator_date:
        log_message(f"WARNING: No new trading data available", log_file)
        log_message(f"  Latest price date: {latest_price_date}", log_file)
        log_message(f"  Latest indicator date: {latest_indicator_date}", log_file)
        log_message(f"  This may be a holiday or weekend. Skipping indicator calculation.", log_file)
        log_message("Status: SKIPPED - No new trading data", log_file)
        return True  # Not an error, just no new data
    
    log_message(f"New trading data confirmed: {latest_price_date}", log_file)
    
    # Step 3: Calculate indicators using master procedure (NO TRIGGER)
    log_message("\nStep 3: Calculating indicators using master procedure...", log_file)
    log_message("  Calling usp_master_update_all_indicators()...", log_file)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        indicator_start = datetime.now()
        cursor.execute("CALL usp_master_update_all_indicators()")
        conn.commit()
        indicator_end = datetime.now()
        indicator_duration = indicator_end - indicator_start
        
        cursor.close()
        conn.close()
        
        log_message(f"  ✅ Indicators calculated successfully in {indicator_duration}", log_file)
        
    except Exception as e:
        log_message(f"  ❌ Error calculating indicators: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return False
    
    # Step 3.5: Refresh optimized views and tables
    log_message("\nStep 3.5: Refreshing optimized signal views...", log_file)
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        log_message("  - Refreshing materialized view...", log_file)
        cursor.execute("REFRESH MATERIALIZED VIEW mv_all_signals")
        conn.commit()
        
        log_message("  - Refreshing latest BUY signals table...", log_file)
        cursor.execute("SELECT refresh_latest_buy_signals()")
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM latest_buy_signals")
        signal_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        log_message(f"  ✅ Optimized views refreshed: {signal_count:,} latest BUY signals", log_file)
        
    except Exception as e:
        log_message(f"  ⚠️  Warning: Could not refresh optimized views: {e}", log_file)
        log_message("  (This is OK if you haven't run database/setup_optimized_architecture.py yet)", log_file)
    
    # Step 4: Verify calculation results
    log_message("\nStep 4: Verifying calculation results...", log_file)
    
    # Get latest date
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(trade_date) FROM daily_prices")
    latest_date = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    verify_success = verify_results(latest_date, log_file)
    
    # Step 5: Export all BUY signals to CSV (incremental)
    log_message("\nStep 5: Updating BUY signals CSV...", log_file)
    export_success = export_all_buy_signals(log_file, latest_date)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    log_message("\n" + "="*70, log_file)
    log_message("AUTOMATION COMPLETED", log_file)
    log_message("="*70, log_file)
    log_message(f"End Time: {end_time}", log_file)
    log_message(f"Duration: {duration}", log_file)
    
    if verify_success and export_success:
        log_message("Status: ✅ SUCCESS - All data verified, complete, and exported", log_file)
        print("\n✅ Daily automation completed successfully!")
        return True
    elif verify_success:
        log_message("Status: ⚠️  PARTIAL SUCCESS - Data verified but export failed", log_file)
        print("\n⚠️  Daily automation completed with warnings - Check logs")
        return True
    else:
        log_message("Status: ❌ FAILED - Verification failed, data may be incomplete", log_file)
        print("\n❌ Daily automation FAILED - Check logs for details")
        return False

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    success = run_daily_automation()
    sys.exit(0 if success else 1)
