from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import psycopg2
import psycopg2.extras
from psycopg2 import pool
import sys
import os
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONN

app = FastAPI(title="NSE Stock Analysis API - Optimized")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================================================
# CONNECTION POOLING - CRITICAL OPTIMIZATION
# =========================================================
# Initialize connection pool (increased to match workers)
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=10,
    maxconn=80,  # Increased to support 30 workers + API requests
    **DB_CONN
)

def get_db():
    """Get connection from pool"""
    return connection_pool.getconn()

def return_db(conn):
    """Return connection to pool"""
    connection_pool.putconn(conn)

# Thread pool for parallel execution - Increased for faster processing
executor = ThreadPoolExecutor(max_workers=30)  # Increased for better parallelism

# =========================================================
# INDICATOR CONFIG (SINGLE SOURCE OF TRUTH)
# =========================================================
SMA_INDICATORS = ['SMA5','SMA10','SMA20','SMA50','SMA100','SMA200']
RSI_INDICATORS = ['RSI7','RSI14','RSI21','RSI50','RSI80']
BB_INDICATORS = [
    'BB10_Upper','BB10_Middle','BB10_Lower',
    'BB20_Upper','BB20_Middle','BB20_Lower',
    'BB50_Upper','BB50_Middle','BB50_Lower',
    'BB100_Upper','BB100_Middle','BB100_Lower'
]
MACD_INDICATORS = ['Short','Long','Standard']
STOCH_INDICATORS = ['STOCH5','STOCH9','STOCH14','STOCH21','STOCH50']

ALL_SIGNAL_INDICATORS = (
    SMA_INDICATORS + RSI_INDICATORS + BB_INDICATORS + 
    MACD_INDICATORS + STOCH_INDICATORS
)

# =========================================================
# INDICATOR CONFIG (SINGLE SOURCE OF TRUTH)
# =========================================================
def _analyze_single_indicator_optimized(
    cur, 
    symbol: str, 
    indicator: str, 
    target: float, 
    days: int,
    request_cache: Dict[str, Tuple[List, List]] = None,
    include_details: bool = False  # NEW: Skip details for faster processing
) -> dict:
    """
    OPTIMIZED VERSION with request-scoped caching
    - Cache is passed per request and cleared after
    - No cross-request caching
    - Still avoids redundant queries within a request
    """
    
    # ------------------------------------------
    # 1. Determine table & columns (unchanged)
    # ------------------------------------------
    table_config = {
        'SMA': ('smatbl', 'indicator', 'value', indicator),
        'RSI': ('rsitbl', 'indicator', 'value', indicator),
        'BB': ('bbtbl', 'indicator', 'value', indicator),
        'MACD': ('macdtbl', 'indicator_set', 'macd_line', indicator),
        'STOCH': ('stochtbl', 'indicator', 'k_value', indicator)
    }
    
    # Determine indicator type
    if indicator.startswith('SMA'):
        config = table_config['SMA']
    elif indicator.startswith('RSI'):
        config = table_config['RSI']
    elif indicator.startswith('BB'):
        config = table_config['BB']
    elif indicator in ['Short', 'Long', 'Standard']:
        config = table_config['MACD']
    elif indicator.startswith('STOCH'):
        config = table_config['STOCH']
    else:
        return {"error": "Unknown indicator type"}
    
    table, indicator_col, value_col, indicator_value = config

    # ------------------------------------------
    # 2. Fetch BUY signals - OPTIMIZED with index
    # ------------------------------------------
    cur.execute(f"""
        SELECT trade_date
        FROM {table}
        WHERE symbol = %s
          AND {indicator_col} = %s
          AND signal = 'BUY'
        ORDER BY trade_date
    """, (symbol, indicator_value))

    buy_dates = [row[0] for row in cur.fetchall()]
    total_buy_signals = len(buy_dates)

    if total_buy_signals == 0:
        return {
            "symbol": symbol,
            "indicator": indicator,
            "totalSignals": 0,
            "completedTrades": 0,
            "openTrades": 0,
            "successful": 0,
            "successRate": 0,
            "avgMaxProfit": 0,
            "avgMaxLoss": 0
        }

    # ------------------------------------------
    # 3. Fetch price data - USE REQUEST CACHE
    # ------------------------------------------
    if request_cache is not None and symbol in request_cache:
        # Use cached data from this request
        price_dates, price_values = request_cache[symbol]
    else:
        # Query database
        cur.execute("""
            SELECT trade_date, close_price
            FROM daily_prices
            WHERE symbol = %s
            ORDER BY trade_date
        """, (symbol,))
        rows = cur.fetchall()

        price_dates = [r[0] for r in rows]
        price_values = [float(r[1]) for r in rows]
        
        # Store in request cache if provided
        if request_cache is not None:
            request_cache[symbol] = (price_dates, price_values)

    # Convert to numpy for vectorized operations
    price_array = np.array(price_values, dtype=np.float64)
    
    # Map date → index for O(1) lookup
    date_index = {d: i for i, d in enumerate(price_dates)}

    # ------------------------------------------
    # 4. Backtesting - VECTORIZED O(N) APPROACH
    # ------------------------------------------
    completed_trades = []
    open_trades = 0
    details = [] if include_details else None  # Only create if needed
    
    target_multiplier = 1.0 + (target / 100.0)

    for trade_date in buy_dates:
        if trade_date not in date_index:
            continue

        entry_idx = date_index[trade_date]
        entry_price = price_array[entry_idx]
        target_price = entry_price * target_multiplier

        # Calculate end index
        end_idx = min(entry_idx + days + 1, len(price_array))
        
        # Slice future prices (single operation, no loop)
        future_prices = price_array[entry_idx + 1:end_idx]
        
        # CRITICAL FIX: Check if we have enough data to complete the analysis
        # If we don't have the full requested days, it's an OPEN trade
        actual_days_available = len(future_prices)
        has_full_window = (entry_idx + days + 1) <= len(price_array)
        
        if actual_days_available == 0:
            # No future data available - this is an OPEN trade
            open_trades += 1
            if include_details:
                details.append({
                    "buyDate": trade_date.isoformat(),
                    "buyPrice": round(float(entry_price), 2),
                    "targetPrice": round(float(target_price), 2),
                    "maxPriceReached": None,
                    "daysChecked": 0,
                    "result": "OPEN"
                })
            continue
        
        # Vectorized operations - find cumulative max
        # IMPORTANT: Include entry price to ensure max >= entry always
        all_prices = np.concatenate([[entry_price], future_prices])
        cummax_all = np.maximum.accumulate(all_prices)
        cummax_prices = cummax_all[1:]  # Skip first element (entry price itself)
        
        # Calculate profit percentages vectorized
        profit_pcts = ((cummax_prices - entry_price) / entry_price) * 100
        
        # Find first index where target is hit
        hit_indices = np.where(profit_pcts >= target)[0]
        
        # Max price reached (always >= entry price due to including entry in cummax)
        max_price_reached = float(cummax_prices[-1])
        max_profit_pct = ((max_price_reached - entry_price) / entry_price) * 100
        days_checked = actual_days_available
        
        if len(hit_indices) > 0:
            # Target HIT - SUCCESS
            exit_idx = hit_indices[0]
            exit_price = cummax_prices[exit_idx]
            exit_profit = ((exit_price - entry_price) / entry_price) * 100
            
            # Store the actual maximum profit reached (not just exit profit)
            completed_trades.append(max_profit_pct)
            
            if include_details:
                details.append({
                    "buyDate": trade_date.isoformat(),
                    "buyPrice": round(float(entry_price), 2),
                    "targetPrice": round(float(target_price), 2),
                    "maxPriceReached": round(float(max_price_reached), 2),
                    "daysChecked": int(exit_idx + 1),
                    "result": "SUCCESS"
                })
        else:
            # Target NOT hit within window
            # CRITICAL FIX: If we don't have the full window, it's OPEN, not FAIL
            if not has_full_window:
                # Insufficient data - this is an OPEN trade
                open_trades += 1
                if include_details:
                    details.append({
                        "buyDate": trade_date.isoformat(),
                        "buyPrice": round(float(entry_price), 2),
                        "targetPrice": round(float(target_price), 2),
                        "maxPriceReached": round(float(max_price_reached), 2),
                        "daysChecked": days_checked,
                        "result": "OPEN"
                    })
            else:
                # Full window available but target not hit - FAIL
                completed_trades.append(max_profit_pct)
                
                if include_details:
                    details.append({
                        "buyDate": trade_date.isoformat(),
                        "buyPrice": round(float(entry_price), 2),
                        "targetPrice": round(float(target_price), 2),
                        "maxPriceReached": round(max_price_reached, 2),
                        "daysChecked": days_checked,
                        "result": "FAIL"
                    })

    # ------------------------------------------
    # 5. Stats calculation
    # ------------------------------------------
    completed_count = len(completed_trades)
    successful = sum(1 for p in completed_trades if p >= target)

    success_rate = (successful / completed_count * 100) if completed_count else 0

    # Calculate average max profit from successful trades only
    successful_trades = [p for p in completed_trades if p >= target]
    avg_max_profit = (sum(successful_trades) / len(successful_trades)) if successful_trades else 0
    
    # Calculate average max loss from failed trades only
    failed_trades = [p for p in completed_trades if p < target]
    avg_max_loss = (sum(failed_trades) / len(failed_trades)) if failed_trades else 0

    result = {
        "symbol": symbol,
        "indicator": indicator,
        "totalSignals": total_buy_signals,
        "completedTrades": completed_count,
        "openTrades": open_trades,
        "successful": successful,
        "successRate": round(success_rate, 2),
        "avgMaxProfit": round(avg_max_profit, 2),
        "avgMaxLoss": round(avg_max_loss, 2),
        "targetPct": target,
        "days": days
    }
    
    # Only include details if requested
    if include_details and details is not None:
        result["details"] = details
    
    return result

# =========================================================
# BATCH PRICE LOADER - REDUCE DB QUERIES
# =========================================================
def _batch_load_prices(cur, symbols: List[str]) -> Dict[str, Tuple[List, List]]:
    """
    Load prices for multiple symbols in a single query
    CRITICAL OPTIMIZATION: Reduces N queries to 1 query
    Uses dict cursor for faster processing
    """
    if not symbols:
        return {}
    
    result = {}
    
    # Fetch all symbols in ONE query with optimized ordering
    cur.execute("""
        SELECT symbol, trade_date, close_price
        FROM daily_prices
        WHERE symbol = ANY(%s)
        ORDER BY symbol, trade_date
    """, (symbols,))
    
    # Use fetchall for faster bulk retrieval
    rows = cur.fetchall()
    
    # Group by symbol using dict for O(1) lookups
    current_symbol = None
    current_dates = []
    current_prices = []
    
    for symbol, trade_date, close_price in rows:
        if symbol != current_symbol:
            # Save previous symbol
            if current_symbol:
                result[current_symbol] = (current_dates, current_prices)
            
            # Start new symbol
            current_symbol = symbol
            current_dates = [trade_date]
            current_prices = [float(close_price)]
        else:
            current_dates.append(trade_date)
            current_prices.append(float(close_price))
    
    # Save last symbol
    if current_symbol:
        result[current_symbol] = (current_dates, current_prices)
    
    return result

# =========================================================
# PARALLEL ANALYSIS WORKER
# =========================================================
def _analyze_worker(args):
    """Worker function for parallel analysis with request-scoped cache"""
    symbol, indicator, target, days, prices_data, request_cache = args
    
    conn = get_db()
    try:
        cur = conn.cursor()
        
        # Skip details for faster processing in bulk analysis
        result = _analyze_single_indicator_optimized(
            cur, symbol, indicator, target, days, request_cache, include_details=False
        )
        return result
    finally:
        cur.close()
        return_db(conn)

# =========================================================
# OPTIMIZED ANALYZE-ALL ENDPOINT
# =========================================================
@app.get("/api/analyze-all")
def analyze_all_signals_optimized(
    target: float = Query(5.0, description="Target profit percentage"),
    days: int = Query(30, description="Days to hold position"),
    limit: int = Query(50, description="Limit number of signals to analyze"),
    parallel: bool = Query(True, description="Use parallel processing")
):
    """
    OPTIMIZED VERSION - Key improvements:
    1. Batch loads all prices in single query (N+1 → 1 query)
    2. Parallel processing using ThreadPoolExecutor
    3. Connection pooling
    4. Non-blocking execution
    """
    start_time = time.time()
    conn = get_db()
    
    try:
        cur = conn.cursor()
        
        # Get all current BUY signals
        cur.execute("""
            SELECT symbol, indicator
            FROM latest_buy_signals
            ORDER BY symbol, indicator
            LIMIT %s
        """, (limit,))
        
        signals = cur.fetchall()
        
        if not signals:
            return {
                "message": "No BUY signals found",
                "total_signals": 0,
                "analyzed": 0,
                "results": []
            }
        
        # Extract unique symbols
        unique_symbols = list(set(s[0] for s in signals))
        
        # CRITICAL: Batch load all prices in ONE query
        batch_start = time.time()
        prices_data = _batch_load_prices(cur, unique_symbols)
        batch_time = time.time() - batch_start
        
        cur.close()
        
        # Parallel or sequential processing
        if parallel and len(signals) > 5:
            # Prepare work items with request cache (prices_data)
            work_items = [
                (symbol, indicator, target, days, prices_data, prices_data)
                for symbol, indicator in signals
            ]
            
            # Execute in parallel
            parallel_start = time.time()
            results = list(executor.map(_analyze_worker, work_items))
            parallel_time = time.time() - parallel_start
        else:
            # Sequential processing (for small batches) - use cached prices
            results = []
            cur = conn.cursor()
            for symbol, indicator in signals:
                result = _analyze_single_indicator_optimized(
                    cur, symbol, indicator, target, days, prices_data
                )
                results.append(result)
            cur.close()
            parallel_time = 0
        
        # Calculate summary statistics
        total_analyzed = len(results)
        avg_success_rate = sum(r['successRate'] for r in results) / total_analyzed if total_analyzed > 0 else 0
        
        elapsed = time.time() - start_time
        
        return {
            "message": "Analysis complete",
            "total_signals": len(signals),
            "analyzed": total_analyzed,
            "target_profit": target,
            "days_to_hold": days,
            "avg_success_rate": round(avg_success_rate, 2),
            "performance": {
                "total_time_seconds": round(elapsed, 2),
                "batch_load_time": round(batch_time, 2),
                "analysis_time": round(parallel_time, 2) if parallel else round(elapsed - batch_time, 2),
                "parallel_processing": parallel
            },
            "results": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "results": []}
    finally:
        return_db(conn)


# =========================================================
# SINGLE INDICATOR ANALYSIS ENDPOINT
# =========================================================
@app.get("/api/analyze")
def analyze_indicator(
    symbol: str = Query(...),
    indicator: str = Query(...),
    target: float = Query(5.0),
    days: int = Query(30)
):
    """Analyze historical performance of an indicator for a symbol"""
    conn = get_db()
    try:
        cur = conn.cursor()
        # Include details for single symbol analysis (user wants to see trade history)
        return _analyze_single_indicator_optimized(cur, symbol, indicator, target, days, None, include_details=True)
    except Exception as e:
        return {"error": str(e)}
    finally:
        cur.close()
        return_db(conn)

# =========================================================
# ANALYZE BY INDICATOR TYPE
# =========================================================
@app.get("/api/analyze-by-type")
def analyze_by_indicator_type(
    indicator_type: str = Query(..., description="Indicator type: SMA, RSI, BB, MACD, STOCH"),
    target: float = Query(5.0),
    days: int = Query(30),
    limit: int = Query(50)
):
    """Analyze all BUY signals for a specific indicator type"""
    conn = get_db()
    
    try:
        cur = conn.cursor()
        
        # Build WHERE clause
        type_map = {
            'SMA': "indicator LIKE 'SMA%'",
            'RSI': "indicator LIKE 'RSI%'",
            'BB': "indicator LIKE 'BB%'",
            'MACD': "indicator IN ('Short', 'Long', 'Standard')",
            'STOCH': "indicator LIKE 'STOCH%'"
        }
        
        where_clause = type_map.get(indicator_type.upper())
        if not where_clause:
            return {"error": f"Unknown indicator type: {indicator_type}"}
        
        # Get signals
        cur.execute(f"""
            SELECT symbol, indicator
            FROM latest_buy_signals
            WHERE {where_clause}
            ORDER BY symbol, indicator
            LIMIT %s
        """, (limit,))
        
        signals = cur.fetchall()
        
        if not signals:
            return {
                "message": f"No BUY signals found for {indicator_type}",
                "indicator_type": indicator_type,
                "total_signals": 0,
                "results": []
            }
        
        # Batch load prices
        unique_symbols = list(set(s[0] for s in signals))
        prices_data = _batch_load_prices(cur, unique_symbols)
        
        # Analyze
        results = []
        for symbol, indicator in signals:
            result = _analyze_single_indicator_optimized(
                cur, symbol, indicator, target, days, None
            )
            results.append(result)
        
        # Summary
        total_analyzed = len(results)
        avg_success_rate = sum(r['successRate'] for r in results) / total_analyzed if total_analyzed > 0 else 0
        
        return {
            "indicator_type": indicator_type,
            "total_signals": len(signals),
            "analyzed": total_analyzed,
            "target_profit": target,
            "days_to_hold": days,
            "avg_success_rate": round(avg_success_rate, 2),
            "results": results
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        cur.close()
        return_db(conn)

# =========================================================
# ANALYZE POWER SIGNALS
# =========================================================
@app.get("/api/analyze-power-signals")
def analyze_power_signals(
    min_signals: int = Query(3, ge=2, le=10),
    target: float = Query(5.0),
    days: int = Query(30),
    limit: int = Query(20)
):
    """Analyze stocks with multiple BUY signals (power signals)"""
    conn = get_db()
    
    try:
        cur = conn.cursor()
        
        # Get power signals
        cur.execute("""
            SELECT 
                symbol,
                COUNT(*) as signal_count,
                ARRAY_AGG(indicator ORDER BY indicator) as indicators
            FROM latest_buy_signals
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY signal_count DESC
            LIMIT %s
        """, (min_signals, limit))
        
        power_stocks = cur.fetchall()
        
        if not power_stocks:
            return {
                "message": f"No stocks found with {min_signals}+ signals",
                "min_signals": min_signals,
                "total_stocks": 0,
                "results": []
            }
        
        # Batch load prices for all symbols
        unique_symbols = [row[0] for row in power_stocks]
        prices_data = _batch_load_prices(cur, unique_symbols)
        
        # Analyze each power signal stock
        results = []
        
        for symbol, signal_count, indicators in power_stocks:
            stock_results = []
            for indicator in indicators:
                result = _analyze_single_indicator_optimized(
                    cur, symbol, indicator, target, days, None
                )
                stock_results.append(result)
            
            # Aggregate metrics
            avg_success_rate = sum(r['successRate'] for r in stock_results) / len(stock_results) if stock_results else 0
            total_signals = sum(r['totalSignals'] for r in stock_results)
            
            results.append({
                "symbol": symbol,
                "signal_count": signal_count,
                "indicators": indicators,
                "avg_success_rate": round(avg_success_rate, 2),
                "total_historical_signals": total_signals,
                "indicator_results": stock_results
            })
        
        # Overall summary
        overall_avg = sum(r['avg_success_rate'] for r in results) / len(results) if results else 0
        
        return {
            "message": "Power signals analysis complete",
            "min_signals": min_signals,
            "total_stocks": len(results),
            "target_profit": target,
            "days_to_hold": days,
            "overall_avg_success_rate": round(overall_avg, 2),
            "results": results
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        cur.close()
        return_db(conn)

# =========================================================
# EXISTING ENDPOINTS (UNCHANGED - JUST USE POOLED CONNECTIONS)
# =========================================================

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/symbol/{symbol}", response_class=HTMLResponse)
def symbol_page(request: Request, symbol: str):
    return templates.TemplateResponse("symbol.html", {"request": request, "symbol": symbol})

@app.get("/diagnostic", response_class=HTMLResponse)
def diagnostic_page(request: Request):
    return templates.TemplateResponse("diagnostic.html", {"request": request})

@app.get("/api/symbols")
def get_symbols(q: str = Query("")):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol
            FROM symbols
            WHERE symbol ILIKE %s
            ORDER BY symbol
        """, (f"%{q.upper()}%",))
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/summary")
def signal_summary():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(trade_date) FROM latest_buy_signals")
        result = cur.fetchone()
        buy_count = result[0] or 0
        latest_date = result[1]

        cur.execute("SELECT COUNT(*) FROM symbols")
        total_symbols = cur.fetchone()[0]

        return {
            "date": latest_date.isoformat() if latest_date else None,
            "total_symbols": total_symbols,
            "buy": buy_count,
            "sell": 0
        }
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/signals/by-indicator")
def signals_by_indicator():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                indicator, 
                COUNT(*) as count,
                MAX(trade_date) as latest_date
            FROM latest_buy_signals
            GROUP BY indicator
            ORDER BY indicator
        """)
        
        rows = cur.fetchall()
        
        if not rows:
            return {"date": None, "indicators": {}}
        
        result = {
            "date": rows[0][2].isoformat() if rows else None,
            "indicators": {}
        }
        
        for indicator, count, _ in rows:
            result["indicators"][indicator] = count
        
        return result
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/signals")
def latest_signals(indicator: str | None = Query(None)):
    start_time = time.time()
    conn = get_db()
    try:
        cur = conn.cursor()
        
        if indicator:
            sql = """
                SELECT symbol, trade_date, indicator, value, signal
                FROM latest_buy_signals
                WHERE indicator = %s
                ORDER BY symbol
            """
            cur.execute(sql, (indicator,))
        else:
            sql = """
                SELECT symbol, trade_date, indicator, value, signal
                FROM latest_buy_signals
                ORDER BY indicator, symbol
            """
            cur.execute(sql)
        
        query_time = time.time() - start_time
        
        results = []
        for r in cur.fetchall():
            indicator_name = r[2]
            if indicator_name in ['Long', 'Short', 'Standard']:
                indicator_name = f"MACD_{indicator_name}"
            
            results.append({
                "symbol": r[0],
                "date": r[1].isoformat(),
                "indicator": indicator_name,
                "value": round(float(r[3]), 2) if r[3] else None,
                "signal": r[4]
            })
        
        total_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] /api/signals - Query: {query_time*1000:.2f}ms, Total: {total_time*1000:.2f}ms, Results: {len(results)}")
        
        return results
        
    except Exception as e:
        print(f"Signals error: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/symbol/{symbol}/indicators")
def symbol_indicators(symbol: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        results = []
        
        # SMA
        cur.execute("""
            SELECT trade_date, indicator, value, signal
            FROM smatbl
            WHERE symbol = %s
            ORDER BY trade_date DESC
            LIMIT 1000
        """, (symbol,))
        
        for r in cur.fetchall():
            results.append({
                "date": r[0].isoformat(),
                "indicator": r[1],
                "value": round(float(r[2]), 2) if r[2] else None,
                "signal": r[3],
                "type": "SMA"
            })
        
        # RSI
        cur.execute("""
            SELECT trade_date, indicator, value, signal
            FROM rsitbl
            WHERE symbol = %s
            ORDER BY trade_date DESC
            LIMIT 1000
        """, (symbol,))
        
        for r in cur.fetchall():
            results.append({
                "date": r[0].isoformat(),
                "indicator": r[1],
                "value": round(float(r[2]), 2) if r[2] else None,
                "signal": r[3],
                "type": "RSI"
            })
        
        # BB
        cur.execute("""
            SELECT trade_date, indicator, value, signal
            FROM bbtbl
            WHERE symbol = %s
            ORDER BY trade_date DESC
            LIMIT 1000
        """, (symbol,))
        
        for r in cur.fetchall():
            results.append({
                "date": r[0].isoformat(),
                "indicator": r[1],
                "value": round(float(r[2]), 2) if r[2] else None,
                "signal": r[3],
                "type": "BB"
            })
        
        # MACD
        cur.execute("""
            SELECT trade_date, indicator_set, macd_line, signal
            FROM macdtbl
            WHERE symbol = %s
            ORDER BY trade_date DESC
            LIMIT 1000
        """, (symbol,))
        
        for r in cur.fetchall():
            results.append({
                "date": r[0].isoformat(),
                "indicator": r[1],
                "value": round(float(r[2]), 2) if r[2] else None,
                "signal": r[3],
                "type": "MACD"
            })
        
        # Stochastic
        cur.execute("""
            SELECT trade_date, indicator, k_value, signal
            FROM stochtbl
            WHERE symbol = %s
            ORDER BY trade_date DESC
            LIMIT 1000
        """, (symbol,))
        
        for r in cur.fetchall():
            results.append({
                "date": r[0].isoformat(),
                "indicator": r[1],
                "value": round(float(r[2]), 2) if r[2] else None,
                "signal": r[3],
                "type": "STOCH"
            })
        
        return sorted(results, key=lambda x: x['date'], reverse=True)
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/symbol/{symbol}/chart")
def symbol_chart(
    symbol: str,
    sma: str = Query(None),
    rsi: str = Query(None),
    bb: str = Query(None),
    macd: str = Query(None),
    stoch: str = Query(None)
):
    """
    Get chart data for a symbol with optional indicator selection
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        
        print(f"[CHART API] Symbol: {symbol}, SMA: {sma}, RSI: {rsi}, BB: {bb}, MACD: {macd}, STOCH: {stoch}")
        
        # Build dynamic query based on selected indicators
        query = """
            SELECT
                dp.trade_date,
                dp.close_price,
                dp.open_price,
                dp.high_price,
                dp.low_price,
                dp.volume
        """
        
        joins = []
        
        # Add SMA if selected
        if sma:
            query += ", sma.value AS sma_value, sma.signal AS sma_signal"
            joins.append(f"LEFT JOIN smatbl sma ON sma.symbol = dp.symbol AND sma.trade_date = dp.trade_date AND sma.indicator = '{sma}'")
        else:
            query += ", NULL AS sma_value, NULL AS sma_signal"
        
        # Add RSI if selected
        if rsi:
            query += ", rsi.value AS rsi_value, rsi.signal AS rsi_signal"
            joins.append(f"LEFT JOIN rsitbl rsi ON rsi.symbol = dp.symbol AND rsi.trade_date = dp.trade_date AND rsi.indicator = '{rsi}'")
        else:
            query += ", NULL AS rsi_value, NULL AS rsi_signal"
        
        # Add Bollinger Bands if selected
        if bb:
            # Extract period and type from BB indicator (e.g., BB20_Lower -> period=20, type=Lower)
            bb_parts = bb.split('_')
            bb_period = bb_parts[0].replace('BB', '')
            bb_type = bb_parts[1] if len(bb_parts) > 1 else 'Middle'
            
            query += ", bb_u.value AS bb_upper, bb_m.value AS bb_middle, bb_l.value AS bb_lower"
            
            # Get signal from the selected band type
            if bb_type == 'Upper':
                query += ", bb_u.signal AS bb_signal"
            elif bb_type == 'Lower':
                query += ", bb_l.signal AS bb_signal"
            else:  # Middle
                query += ", bb_m.signal AS bb_signal"
            
            joins.append(f"LEFT JOIN bbtbl bb_u ON bb_u.symbol = dp.symbol AND bb_u.trade_date = dp.trade_date AND bb_u.indicator = 'BB{bb_period}_Upper'")
            joins.append(f"LEFT JOIN bbtbl bb_m ON bb_m.symbol = dp.symbol AND bb_m.trade_date = dp.trade_date AND bb_m.indicator = 'BB{bb_period}_Middle'")
            joins.append(f"LEFT JOIN bbtbl bb_l ON bb_l.symbol = dp.symbol AND bb_l.trade_date = dp.trade_date AND bb_l.indicator = 'BB{bb_period}_Lower'")
            
            print(f"[CHART API] BB indicator: {bb}, period: {bb_period}, type: {bb_type}, getting signal from bb_{bb_type.lower()}")
        else:
            query += ", NULL AS bb_upper, NULL AS bb_middle, NULL AS bb_lower, NULL AS bb_signal"
        
        # Add MACD if selected
        if macd:
            query += ", macd_line.macd_line AS macd_line, macd_line.signal_line AS macd_signal, macd_line.histogram AS macd_histogram, macd_line.signal AS macd_signal_flag"
            joins.append(f"LEFT JOIN macdtbl macd_line ON macd_line.symbol = dp.symbol AND macd_line.trade_date = dp.trade_date AND macd_line.indicator_set = '{macd}'")
            print(f"[CHART API] Adding MACD join for indicator_set: {macd}")
        else:
            query += ", NULL AS macd_line, NULL AS macd_signal, NULL AS macd_histogram, NULL AS macd_signal_flag"
        
        # Add Stochastic if selected
        if stoch:
            query += ", stoch.k_value AS stoch_k, stoch.d_value AS stoch_d, stoch.signal AS stoch_signal"
            joins.append(f"LEFT JOIN stochtbl stoch ON stoch.symbol = dp.symbol AND stoch.trade_date = dp.trade_date AND stoch.indicator = '{stoch}'")
        else:
            query += ", NULL AS stoch_k, NULL AS stoch_d, NULL AS stoch_signal"
        
        query += "\nFROM daily_prices dp\n"
        query += "\n".join(joins)
        query += "\nWHERE dp.symbol = %s\nORDER BY dp.trade_date"
        
        print(f"[CHART API] Executing query...")
        cur.execute(query, (symbol,))
        rows = cur.fetchall()
        
        print(f"[CHART API] Retrieved {len(rows)} rows")
        
        result = [
            {
                "date": r[0].isoformat(),
                "price": float(r[1]),
                "open": float(r[2]) if r[2] else None,
                "high": float(r[3]) if r[3] else None,
                "low": float(r[4]) if r[4] else None,
                "volume": int(r[5]) if r[5] else None,
                "sma": float(r[6]) if r[6] else None,
                "sma_signal": r[7],
                "rsi": float(r[8]) if r[8] else None,
                "rsi_signal": r[9],
                "bb_upper": float(r[10]) if r[10] else None,
                "bb_middle": float(r[11]) if r[11] else None,
                "bb_lower": float(r[12]) if r[12] else None,
                "bb_signal": r[13],
                "macd_line": float(r[14]) if r[14] else None,
                "macd_signal": float(r[15]) if r[15] else None,
                "macd_histogram": float(r[16]) if r[16] else None,
                "macd_signal_flag": r[17],
                "stoch_k": float(r[18]) if r[18] else None,
                "stoch_d": float(r[19]) if r[19] else None,
                "stoch_signal": r[20]
            }
            for r in rows
        ]
        
        if macd and len(result) > 0:
            # Count how many MACD values are not null
            macd_count = sum(1 for r in result if r['macd_line'] is not None)
            print(f"[CHART API] MACD data points: {macd_count}/{len(result)}")
        
        return result
        
    finally:
        cur.close()
        return_db(conn)

# Global cache for indicators (refreshed every 5 minutes)
_indicators_cache = None
_indicators_cache_time = 0
INDICATORS_CACHE_TTL = 300  # 5 minutes

@app.get("/api/indicators")
def indicators_list():
    """Get all available indicators that have BUY signals in the database (cached)"""
    global _indicators_cache, _indicators_cache_time
    
    # Check cache
    current_time = time.time()
    if _indicators_cache and (current_time - _indicators_cache_time) < INDICATORS_CACHE_TTL:
        return _indicators_cache
    
    conn = get_db()
    try:
        cur = conn.cursor()
        
        # Get unique indicators that actually have BUY signals
        indicators = set()
        
        # SMA indicators with BUY signals
        cur.execute("SELECT DISTINCT indicator FROM smatbl WHERE signal = 'BUY' ORDER BY indicator")
        indicators.update([row[0] for row in cur.fetchall()])
        
        # RSI indicators with BUY signals
        cur.execute("SELECT DISTINCT indicator FROM rsitbl WHERE signal = 'BUY' ORDER BY indicator")
        indicators.update([row[0] for row in cur.fetchall()])
        
        # Bollinger Bands indicators with BUY signals
        cur.execute("SELECT DISTINCT indicator FROM bbtbl WHERE signal = 'BUY' ORDER BY indicator")
        indicators.update([row[0] for row in cur.fetchall()])
        
        # MACD indicators with BUY signals (use indicator_set column)
        cur.execute("SELECT DISTINCT indicator_set FROM macdtbl WHERE signal = 'BUY' ORDER BY indicator_set")
        indicators.update([row[0] for row in cur.fetchall()])
        
        # Stochastic indicators with BUY signals
        cur.execute("SELECT DISTINCT indicator FROM stochtbl WHERE signal = 'BUY' ORDER BY indicator")
        indicators.update([row[0] for row in cur.fetchall()])
        
        # Convert to sorted list
        indicator_list = sorted(list(indicators))
        
        # Update cache
        _indicators_cache = indicator_list
        _indicators_cache_time = current_time
        
        print(f"[API] Found {len(indicator_list)} indicators with BUY signals (cached for 5 min)")
        
        return indicator_list
        
    except Exception as e:
        print(f"[API] Error fetching indicators from database: {e}")
        # Fallback to hardcoded list if database query fails
        return ALL_SIGNAL_INDICATORS
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/signals/power")
def power_signals(min_signals: int = Query(3, ge=2, le=10)):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                symbol,
                COUNT(*) as signal_count,
                ARRAY_AGG(indicator ORDER BY indicator) as indicators,
                MAX(trade_date) as trade_date
            FROM latest_buy_signals
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY signal_count DESC, symbol
        """, (min_signals,))
        
        results = []
        for row in cur.fetchall():
            results.append({
                "symbol": row[0],
                "signal_count": row[1],
                "indicators": row[2],
                "date": row[3].isoformat() if row[3] else None
            })
        
        return {
            "min_signals": min_signals,
            "total_stocks": len(results),
            "stocks": results
        }
    finally:
        cur.close()
        return_db(conn)

@app.get("/api/signals/stats")
def signal_statistics():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                CASE 
                    WHEN indicator LIKE 'SMA%' THEN 'SMA'
                    WHEN indicator LIKE 'RSI%' THEN 'RSI'
                    WHEN indicator LIKE 'BB%' THEN 'Bollinger Bands'
                    WHEN indicator IN ('Short', 'Long', 'Standard') THEN 'MACD'
                    WHEN indicator LIKE 'STOCH%' THEN 'Stochastic'
                    ELSE 'Other'
                END as indicator_type,
                COUNT(*) as signal_count,
                COUNT(DISTINCT symbol) as unique_symbols
            FROM latest_buy_signals
            GROUP BY indicator_type
            ORDER BY signal_count DESC
        """)
        
        results = []
        for row in cur.fetchall():
            results.append({
                "type": row[0],
                "signal_count": row[1],
                "unique_symbols": row[2]
            })
        
        return {
            "by_type": results,
            "total_signals": sum(r["signal_count"] for r in results)
        }
    finally:
        cur.close()
        return_db(conn)

# =========================================================
# METRICS ENDPOINT
# =========================================================
@app.get("/api/metrics")
def metrics():
    """Performance and health metrics"""
    return {
        "connection_pool": {
            "max_connections": connection_pool.maxconn,
            "min_connections": connection_pool.minconn,
            "available": len(connection_pool._pool),
            "in_use": connection_pool.maxconn - len(connection_pool._pool)
        },
        "thread_pool": {
            "max_workers": executor._max_workers,
            "queue_size": executor._work_queue.qsize() if hasattr(executor._work_queue, 'qsize') else 0
        }
    }

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        
        # Also check latest date in database
        cur.execute("SELECT MAX(trade_date) FROM daily_prices")
        latest_date = cur.fetchone()[0]
        
        cur.close()
        return_db(conn)
        return {
            "status": "healthy", 
            "database": "connected",
            "latest_price_date": latest_date.isoformat() if latest_date else None
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# =========================================================
# PROGRESSIVE ANALYSIS - SHOW FIRST 50 IMMEDIATELY
# =========================================================
@app.get("/api/analyze-progressive")
def analyze_progressive(
    target: float = Query(5.0, description="Target profit percentage"),
    days: int = Query(30, description="Days to hold position"),
    batch_size: int = Query(50, ge=10, le=10000, description="Batch size (10-10000)"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    PROGRESSIVE LOADING - Returns results in batches
    - First call (offset=0): Analyzes ALL signals, sorts them, returns first 50
    - Subsequent calls: Returns next batch from already-sorted results
    - Frontend can show results immediately and load more in background
    
    IMPORTANT: First batch analyzes everything and caches sorted results
    """
    global _cached_analysis_results  # Declare at the top of function
    
    start_time = time.time()
    conn = get_db()
    
    # Create request-scoped cache
    request_cache = {}
    
    try:
        cur = conn.cursor()
        
        # Get total count first
        cur.execute("SELECT COUNT(*) FROM latest_buy_signals")
        total_signals = cur.fetchone()[0]
        
        # FIRST BATCH: Analyze ALL signals and sort
        if offset == 0:
            print(f"[PROGRESSIVE] First batch - analyzing ALL {total_signals} signals...")
            
            # Get ALL signals
            cur.execute("""
                SELECT symbol, indicator
                FROM latest_buy_signals
                ORDER BY symbol, indicator
            """)
            
            all_signals = cur.fetchall()
            
            # Extract unique symbols
            unique_symbols = list(set(symbol for symbol, _ in all_signals))
            print(f"[PROGRESSIVE] Loading prices for {len(unique_symbols)} symbols...")
            
            # Batch load ALL prices
            prices_data = _batch_load_prices(cur, unique_symbols)
            request_cache.update(prices_data)
            
            cur.close()
            return_db(conn)
            
            # Analyze ALL signals in parallel
            print(f"[PROGRESSIVE] Analyzing {len(all_signals)} signals...")
            work_items = [
                (symbol, indicator, target, days, None, request_cache)
                for symbol, indicator in all_signals
            ]
            
            chunksize = max(1, len(work_items) // 30)
            all_results = list(executor.map(_analyze_worker, work_items, chunksize=chunksize))
            
            # Sort ALL results by success rate (GLOBAL SORT)
            all_results.sort(key=lambda x: (-x.get('successRate', 0), x.get('symbol', '')))
            print(f"[PROGRESSIVE] Sorted {len(all_results)} results by success rate")
            
            # Cache sorted results in memory (using a simple in-memory cache)
            # In production, you'd use Redis or similar
            _cached_analysis_results = {
                'results': all_results,
                'target': target,
                'days': days,
                'timestamp': time.time()
            }
            
            # Return first batch
            first_batch = all_results[0:batch_size]
            total_time = time.time() - start_time
            
            print(f"[PROGRESSIVE] Returning first {len(first_batch)} results (total time: {total_time:.2f}s)")
            
            return {
                "message": "First batch complete (all analyzed and sorted)",
                "total_signals": total_signals,
                "batch_size": len(first_batch),
                "offset": 0,
                "has_more": batch_size < len(all_results),
                "next_offset": batch_size,
                "target_profit": target,
                "days_to_hold": days,
                "processing_time_seconds": round(total_time, 2),
                "results": first_batch
            }
        
        # SUBSEQUENT BATCHES: Return from cached sorted results
        else:
            print(f"[PROGRESSIVE] Returning batch at offset {offset}...")
            
            # Get cached results
            if _cached_analysis_results is None:
                return {
                    "error": "No cached results. Please start from offset 0.",
                    "results": []
                }
            
            cached = _cached_analysis_results
            all_results = cached['results']
            
            # Return batch from cached sorted results
            end_offset = offset + batch_size
            batch = all_results[offset:end_offset]
            
            total_time = time.time() - start_time
            
            return {
                "message": "Batch from cache",
                "total_signals": len(all_results),
                "batch_size": len(batch),
                "offset": offset,
                "has_more": end_offset < len(all_results),
                "next_offset": end_offset if end_offset < len(all_results) else None,
                "target_profit": cached['target'],
                "days_to_hold": cached['days'],
                "processing_time_seconds": round(total_time, 2),
                "results": batch
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "results": []}
    finally:
        request_cache.clear()
        try:
            return_db(conn)
        except:
            pass


# Global cache for progressive loading (simple in-memory cache)
_cached_analysis_results = None


# =========================================================
# FAST ANALYSIS - ALL SIGNALS AT ONCE
# =========================================================
@app.get("/api/analyze-fast")
def analyze_fast(
    target: float = Query(5.0, description="Target profit percentage"),
    days: int = Query(30, description="Days to hold position"),
    limit: int = Query(10000, ge=1, le=10000, description="Maximum number of signals to analyze")
):
    """
    ULTRA-FAST VERSION with request-scoped caching
    - Cache exists only for this request
    - Cleared after request completes
    - Avoids redundant queries within request
    """
    start_time = time.time()
    conn = get_db()
    
    # Create request-scoped cache (cleared after this request)
    request_cache = {}
    
    try:
        cur = conn.cursor()
        
        # Get all latest BUY signals (with limit)
        cur.execute("""
            SELECT symbol, indicator
            FROM latest_buy_signals
            ORDER BY symbol, indicator
            LIMIT %s
        """, (limit,))
        
        signals = cur.fetchall()
        
        if not signals:
            cur.close()
            return {
                "message": "No BUY signals found",
                "total_signals": 0,
                "results": []
            }
        
        # Extract unique symbols for batch price loading
        unique_symbols = list(set(symbol for symbol, _ in signals))
        print(f"[FAST] Loading prices for {len(unique_symbols)} unique symbols...")
        
        # Batch load all prices into request cache
        prices_data = _batch_load_prices(cur, unique_symbols)
        
        # Populate request cache
        request_cache.update(prices_data)
        
        cur.close()
        return_db(conn)  # Return connection early
        
        load_time = time.time() - start_time
        print(f"[FAST] Prices loaded in {load_time:.2f}s, cached {len(request_cache)} symbols")
        
        # Batch analyze all signals in parallel with request cache
        work_items = [
            (symbol, indicator, target, days, None, request_cache)
            for symbol, indicator in signals
        ]
        
        print(f"[FAST] Analyzing {len(signals)} signals in parallel with 30 workers...")
        analysis_start = time.time()
        
        # Use chunksize for better load distribution
        chunksize = max(1, len(work_items) // 30)
        results = list(executor.map(_analyze_worker, work_items, chunksize=chunksize))
        
        analysis_time = time.time() - analysis_start
        
        total_time = time.time() - start_time
        print(f"[FAST] Analysis complete in {analysis_time:.2f}s (total: {total_time:.2f}s)")
        
        # Sort by success rate (highest first), then by symbol
        results.sort(key=lambda x: (-x.get('successRate', 0), x.get('symbol', '')))
        
        return {
            "message": "Analysis complete",
            "total_signals": len(signals),
            "analyzed": len(results),
            "target_profit": target,
            "days_to_hold": days,
            "processing_time_seconds": round(total_time, 2),
            "results": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "results": []}
    finally:
        # Clear request cache (important!)
        request_cache.clear()
        
        # Make sure connection is returned
        try:
            return_db(conn)
        except:
            pass


# =========================================================
# GROUPED ANALYSIS - BY COMPANY (AGGREGATE ALL INDICATORS)
# =========================================================
@app.get("/api/analyze-grouped")
def analyze_grouped(
    target: float = Query(5.0, description="Target profit percentage"),
    days: int = Query(30, description="Days to hold position"),
    limit: int = Query(10000, ge=1, le=10000, description="Maximum number of signals to analyze")
):
    """
    Analyze signals grouped by company symbol
    - Aggregates all indicators for each company
    - Shows combined statistics (total signals, success, failure, open, success%)
    - Returns one row per company instead of one row per indicator
    """
    start_time = time.time()
    conn = get_db()
    request_cache = {}
    
    try:
        cur = conn.cursor()
        
        # Get all latest BUY signals
        cur.execute("""
            SELECT symbol, indicator
            FROM latest_buy_signals
            ORDER BY symbol, indicator
            LIMIT %s
        """, (limit,))
        
        signals = cur.fetchall()
        
        if not signals:
            cur.close()
            return {
                "message": "No BUY signals found",
                "total_companies": 0,
                "results": []
            }
        
        # Extract unique symbols for batch price loading
        unique_symbols = list(set(symbol for symbol, _ in signals))
        print(f"[GROUPED] Loading prices for {len(unique_symbols)} unique symbols...")
        
        # Batch load all prices
        prices_data = _batch_load_prices(cur, unique_symbols)
        request_cache.update(prices_data)
        
        cur.close()
        return_db(conn)
        
        # Analyze all signals
        work_items = [
            (symbol, indicator, target, days, None, request_cache)
            for symbol, indicator in signals
        ]
        
        print(f"[GROUPED] Analyzing {len(signals)} signals...")
        chunksize = max(1, len(work_items) // 30)
        results = list(executor.map(_analyze_worker, work_items, chunksize=chunksize))
        
        # Group results by symbol
        grouped = {}
        for result in results:
            symbol = result.get('symbol')
            if not symbol:
                continue
            
            if symbol not in grouped:
                grouped[symbol] = {
                    'symbol': symbol,
                    'indicators': [],
                    'total_signals': 0,
                    'successful': 0,
                    'failed': 0,
                    'open': 0,
                    'completed': 0
                }
            
            grouped[symbol]['indicators'].append(result.get('indicator'))
            grouped[symbol]['total_signals'] += result.get('totalSignals', 0)
            grouped[symbol]['successful'] += result.get('successful', 0)
            grouped[symbol]['failed'] += result.get('failed', 0)
            grouped[symbol]['open'] += result.get('open', 0)
            grouped[symbol]['completed'] += result.get('completedTrades', 0)
        
        # Calculate success rate for each company
        final_results = []
        for symbol, data in grouped.items():
            success_rate = 0
            if data['completed'] > 0:
                success_rate = round((data['successful'] / data['completed']) * 100, 2)
            
            final_results.append({
                'symbol': symbol,
                'indicators': ', '.join(data['indicators']),
                'indicator_count': len(data['indicators']),
                'totalSignals': data['total_signals'],
                'successful': data['successful'],
                'failed': data['failed'],
                'open': data['open'],
                'completedTrades': data['completed'],
                'successRate': success_rate
            })
        
        # Sort by success rate (highest first), then by symbol
        final_results.sort(key=lambda x: (-x.get('successRate', 0), x.get('symbol', '')))
        
        total_time = time.time() - start_time
        
        return {
            "message": "Grouped analysis complete",
            "total_companies": len(final_results),
            "total_signals_analyzed": len(signals),
            "target_profit": target,
            "days_to_hold": days,
            "processing_time_seconds": round(total_time, 2),
            "results": final_results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "results": []}
    finally:
        request_cache.clear()
        try:
            return_db(conn)
        except:
            pass
