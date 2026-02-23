#!/usr/bin/env python3
"""
NSE Bhavcopy Downloader
Downloads complete NSE Bhavcopy file and processes only our 2086 companies
Much faster than individual API calls (minutes vs hours)
"""

import requests
import zipfile
import time
import random
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import sys
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONN

# =====================================
# CONFIG
# =====================================
HOME_URL = "https://www.nseindia.com"
BHAVCOPY_BASE = "https://nsearchives.nseindia.com/content/cm"
DATA_DIR = Path("bhavdata")
DATA_DIR.mkdir(exist_ok=True)

# Configuration options
KEEP_DOWNLOADED_FILES = True  # Set to False to delete files after processing
MAX_FILES_TO_KEEP = 30  # Keep last 30 files, delete older ones

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# =====================================
# DATABASE FUNCTIONS
# =====================================
def get_connection():
    return psycopg2.connect(**DB_CONN)

def get_our_symbols():
    """Get our 2086 company symbols from database"""
    print("Loading our company symbols from database...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT symbol_id, symbol
            FROM symbols
            ORDER BY symbol
        """)

        symbols = cursor.fetchall()
        
        # Strip 'NSE:' prefix from symbols to match Bhavcopy format
        symbol_dict = {}
        for symbol_id, symbol in symbols:
            clean_symbol = symbol.replace('NSE:', '').strip()
            symbol_dict[clean_symbol] = symbol_id
        
        print(f"   Loaded {len(symbol_dict)} symbols from database")
        return symbol_dict
        
    finally:
        cursor.close()
        conn.close()

def cleanup_old_files():
    """Clean up old Bhavcopy files to save disk space"""
    if not KEEP_DOWNLOADED_FILES:
        return
    
    try:
        # Get all ZIP files in bhavdata directory
        zip_files = list(DATA_DIR.glob("*.zip"))
        
        if len(zip_files) <= MAX_FILES_TO_KEEP:
            return
        
        # Sort by modification time (oldest first)
        zip_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Delete oldest files
        files_to_delete = zip_files[:-MAX_FILES_TO_KEEP]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"   Deleted old file: {file_path.name}")
            except:
                pass
                
        if files_to_delete:
            print(f"   Cleaned up {len(files_to_delete)} old files")
            
    except Exception as e:
        print(f"   Warning: Cleanup error: {e}")

def store_price_data(price_data, trade_date):
    if not price_data:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # First, delete existing data for this date
        cursor.execute(
            "DELETE FROM daily_prices WHERE trade_date = %s",
            (trade_date,)
        )

        # Build insert query with symbol_id lookup
        insert_query = """
            INSERT INTO daily_prices
            (symbol_id, symbol, trade_date, open_price, high_price, low_price, close_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Prepare records with symbol_id lookup
        records = []
        failed_records = []
        
        for r in price_data:
            symbol = r["symbol"]
            
            # Look up symbol_id from symbols table
            cursor.execute(
                "SELECT symbol_id FROM symbols WHERE symbol = %s",
                (symbol,)
            )
            result = cursor.fetchone()
            
            if result:
                symbol_id = result[0]
                records.append((
                    symbol_id,
                    symbol,
                    trade_date,
                    r["open_price"],
                    r["high_price"],
                    r["low_price"],
                    r["close_price"],
                    r["volume"]
                ))
            else:
                failed_records.append(symbol)
        
        # Insert valid records
        if records:
            psycopg2.extras.execute_batch(
                cursor,
                insert_query,
                records,
                page_size=1000
            )
        
        conn.commit()
        
        # Report results
        if failed_records:
            print(f"   WARNING: {len(failed_records)} symbols not found in symbols table:")
            for sym in failed_records[:10]:  # Show first 10
                print(f"      - {sym}")
            if len(failed_records) > 10:
                print(f"      ... and {len(failed_records) - 10} more")
        
        return len(records)

    except Exception as e:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


# =====================================
# SESSION CREATION
# =====================================
def create_session():
    """Create robust session with retry logic"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    
    # Warm up session
    try:
        print("Warming up NSE session...")
        response = session.get(HOME_URL, timeout=20)
        if response.status_code == 200:
            print("   NSE session initialized successfully")
        else:
            print(f"   Session warmup returned status: {response.status_code}")
    except Exception as e:
        print(f"   Session warmup warning: {e}")
    
    # Random delay to avoid detection
    time.sleep(random.uniform(1, 3))
    
    return session

# =====================================
# BHAVCOPY DOWNLOAD
# =====================================
def download_bhavcopy(session, target_date=None, max_lookback=10):
    """Download NSE Bhavcopy for specified date or latest available"""
    
    if target_date is None:
        target_date = datetime.now().date()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    
    print(f"Searching for Bhavcopy starting from {target_date}...")
    
    for i in range(max_lookback):
        check_date = target_date - timedelta(days=i)
        
        # Skip weekends (Saturday=5, Sunday=6)
        if check_date.weekday() >= 5:
            continue
            
        # Use the working format: BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip
        date_str = check_date.strftime("%Y%m%d")
        filename = f"BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"
        url = f"{BHAVCOPY_BASE}/{filename}"
        
        print(f"   Trying: {filename} ({check_date})")
        
        try:
            # Add random delay
            time.sleep(random.uniform(0.5, 1.5))
            
            response = session.get(url, timeout=60)
            
            if response.status_code == 200:
                zip_path = DATA_DIR / filename
                
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                
                file_size = len(response.content) / 1024  # KB
                print(f"   Downloaded: {filename} ({file_size:.1f} KB)")
                
                return zip_path, check_date
                
            else:
                print(f"   Not available (Status: {response.status_code})")
                
        except Exception as e:
            print(f"   Error downloading {filename}: {str(e)[:50]}")
    
    print(f"No Bhavcopy found in {max_lookback} day lookback window")
    return None, None

# =====================================
# ZIP EXTRACTION & CSV PROCESSING
# =====================================
def extract_and_process_bhavcopy(zip_path, trade_date, our_symbols):
    """Extract ZIP and process CSV data for our companies only"""
    
    print(f"Extracting and processing: {zip_path.name}")
    
    try:
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            zip_ref.extractall(DATA_DIR)
            
            # Find the CSV file
            csv_file = None
            for file in extracted_files:
                if file.endswith('.csv'):
                    csv_file = DATA_DIR / file
                    break
            
            if not csv_file:
                print("   No CSV file found in ZIP")
                return []
        
        print(f"   Processing CSV: {csv_file.name}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        print(f"   Total records in Bhavcopy: {len(df)}")
        
        # Filter for equity stocks only (exclude bonds, derivatives, etc.)
        equity_df = df[df['Sgmt'] == 'CM'].copy()

        equity_df['series_priority'] = equity_df['SctySrs'].map({
            'EQ': 1,
            'BE': 2,
            'ST': 3,
            'SM': 4,
            'BZ': 5,
            'BL': 6,
            'BT': 7,
            'GC': 8,
            'IL': 9
        })

        equity_df = (
        equity_df
        .sort_values(['TckrSymb', 'series_priority'])
        .drop_duplicates(['TckrSymb'], keep='first')
    )


        print(f"   Equity records: {len(equity_df)}")
        print(f"   Filtering for our {len(our_symbols)} companies...")
        
        # Filter for our symbols only
        filtered_df = equity_df[equity_df['TckrSymb'].isin(our_symbols.keys())]
        
        print(f"   Found {len(filtered_df)} matching records")
        
        # Convert to our format
        price_data = []
        
        for _, row in filtered_df.iterrows():
            try:
                symbol = row['TckrSymb']
                # Add NSE: prefix to match symbols table format
                symbol_with_prefix = f"NSE:{symbol}"
                
                price_record = {
                    'symbol': symbol_with_prefix,
                    'open_price': float(row['OpnPric']),
                    'high_price': float(row['HghPric']),
                    'low_price': float(row['LwPric']),
                    'close_price': float(row['ClsPric']),
                    'volume': float(row['TtlTradgVol']) if pd.notna(row['TtlTradgVol']) else 0
                }
                
                price_data.append(price_record)
                
            except Exception as e:
                print(f"      Error processing {row.get('TckrSymb', 'unknown')}: {e}")
        
        # Clean up extracted CSV files (keep ZIP for audit)
        try:
            csv_file.unlink()  # Delete CSV file (extracted)
            print(f"   Cleaned up extracted CSV file")
        except:
            pass
        
        return price_data
        
    except Exception as e:
        print(f"   Error processing ZIP file: {e}")
        return []

# =====================================
# FILE MANAGEMENT
# =====================================
def cleanup_old_files():
    """Clean up old Bhavcopy files to save disk space"""
    if not KEEP_DOWNLOADED_FILES:
        return
    
    try:
        # Get all ZIP files in bhavdata directory
        zip_files = list(DATA_DIR.glob("*.zip"))
        
        if len(zip_files) <= MAX_FILES_TO_KEEP:
            return
        
        # Sort by modification time (oldest first)
        zip_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Delete oldest files
        files_to_delete = zip_files[:-MAX_FILES_TO_KEEP]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"   Deleted old file: {file_path.name}")
            except:
                pass
                
        if files_to_delete:
            print(f"   Cleaned up {len(files_to_delete)} old files")
            
    except Exception as e:
        print(f"   Warning: Cleanup error: {e}")

# =====================================
# MAIN PIPELINE
# =====================================
def run_bhavcopy_pipeline(target_date=None):
    """Main pipeline to download and process Bhavcopy"""
    
    print("NSE BHAVCOPY PIPELINE STARTED")
    print("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Get our company symbols
        our_symbols = get_our_symbols()
        
        if not our_symbols:
            print("ERROR: No symbols found in database")
            return False
        
        # Step 2: Create session
        session = create_session()
        
        # Step 3: Download Bhavcopy
        zip_path, actual_date = download_bhavcopy(session, target_date)
        
        if not zip_path:
            print("ERROR: Failed to download Bhavcopy")
            return False
        
        # Step 4: Process Bhavcopy
        price_data = extract_and_process_bhavcopy(zip_path, actual_date, our_symbols)
        
        if not price_data:
            print("ERROR: No price data extracted")
            return False
        
        # Step 5: Store in database
        stored_count = store_price_data(price_data, actual_date)
        
        # Step 6: File management
        if KEEP_DOWNLOADED_FILES:
            print(f"   Bhavcopy file saved: {zip_path}")
            print(f"   Files stored in: {DATA_DIR}")
            cleanup_old_files()  # Clean up old files if too many
        else:
            try:
                zip_path.unlink()  # Delete ZIP file
                print("   Cleaned up downloaded files")
            except:
                pass
        
        session.close()
        
        # Final results
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print("BHAVCOPY PIPELINE RESULTS")
        print("=" * 50)
        print(f"Trade Date: {actual_date}")
        print(f"Companies Processed: {stored_count}/{len(our_symbols)}")
        print(f"Duration: {duration}")
        print(f"Success Rate: {(stored_count/len(our_symbols)*100):.1f}%")
        
        if stored_count > 0:
            print("BHAVCOPY PIPELINE COMPLETED SUCCESSFULLY!")
            return True
        else:
            print("BHAVCOPY PIPELINE FAILED - No data stored")
            return False
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE Bhavcopy Downloader')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)', default=None)
    
    args = parser.parse_args()
    
    success = run_bhavcopy_pipeline(args.date)
    
    if success:
        print("\nReady for indicator calculations!")
        print("Run: python pipeline/calculate_indicators.py")
    else:
        print("\nCheck logs and try again")

if __name__ == "__main__":
    main()