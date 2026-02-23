#!/usr/bin/env python3
"""
Export ALL BUY signals to a single CSV file (2016-2026)
Complete historical data in one file - GROUPED BY COMPANY
"""
import psycopg2
from config import DB_CONN
import pandas as pd
from datetime import datetime
import os

def export_all_signals():
    print("="*70)
    print("EXPORTING ALL BUY SIGNALS TO SINGLE CSV FILE")
    print("="*70)
    
    conn = psycopg2.connect(**DB_CONN)
    output_dir = "exports"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_dir}/All_BUY_Signals_2016_2026_{timestamp}.csv"
    
    print(f"\n📁 Output: {csv_file}")
    print("⏳ Exporting all historical BUY signals (2016-2026)...")
    print("   This may take a few minutes...\n")
    
    # Single query to get ALL BUY signals - SORTED BY COMPANY FIRST
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
    
    print("📊 Fetching data from database...")
    df = pd.read_sql_query(query, conn)
    print(f"✓ Retrieved {len(df):,} BUY signals")
    
    print(f"\n💾 Writing to CSV file...")
    df.to_csv(csv_file, index=False)
    conn.close()
    
    # Get file info
    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
    
    # Get date range
    min_date = df['trade_date'].min()
    max_date = df['trade_date'].max()
    
    # Get unique counts
    unique_companies = df['symbol'].nunique()
    unique_indicators = df['indicator'].nunique()
    unique_dates = df['trade_date'].nunique()
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETE!")
    print("="*70)
    print(f"\n📊 File: {csv_file}")
    print(f"💾 Size: {file_size:.2f} MB")
    print(f"\n📈 Data Summary:")
    print(f"   • Total BUY Signals: {len(df):,}")
    print(f"   • Date Range: {min_date} to {max_date}")
    print(f"   • Trading Days: {unique_dates:,}")
    print(f"   • Companies: {unique_companies:,}")
    print(f"   • Indicators: {unique_indicators}")
    print(f"\n📋 Columns:")
    print(f"   1. symbol - Company symbol (e.g., NSE:RELIANCE)")
    print(f"   2. trade_date - Date of BUY signal")
    print(f"   3. indicator - Indicator name (SMA10, RSI14, etc.)")
    print(f"   4. value - Indicator value at signal time")
    print(f"   5. signal - Always 'BUY'")
    print(f"\n✨ Data is GROUPED BY COMPANY!")
    print(f"   All dates for Company 1, then Company 2, etc.")
    print(f"\n💡 Tip: Use Excel filters to analyze specific:")
    print(f"   - Companies (filter by symbol)")
    print(f"   - Indicators (filter by indicator)")
    print(f"   - Time periods (filter by trade_date)")
    print(f"   - Create pivot tables for deeper analysis")

if __name__ == "__main__":
    try:
        export_all_signals()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
