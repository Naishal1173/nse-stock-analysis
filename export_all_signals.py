import psycopg2
import csv
from datetime import datetime
from config import DB_CONN

def export_all_buy_signals():
    """Export all BUY signals from latest_buy_signals table to CSV"""
    
    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()
    
    try:
        # Get all BUY signals
        cur.execute("""
            SELECT 
                symbol,
                date,
                indicator,
                value,
                signal,
                created_at
            FROM latest_buy_signals
            ORDER BY symbol, indicator
        """)
        
        signals = cur.fetchall()
        
        if not signals:
            print("No BUY signals found in latest_buy_signals table")
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/All_BUY_Signals_{timestamp}.csv"
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Symbol', 'Date', 'Indicator', 'Value', 'Signal', 'Created At'])
            
            # Write data
            for row in signals:
                writer.writerow(row)
        
        print(f"✓ Exported {len(signals)} BUY signals to {filename}")
        
    except Exception as e:
        print(f"Error exporting signals: {e}")
    
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    export_all_buy_signals()
