"""
AI Chatbot for NSE Stock Analysis System
Uses Google Gemini API to provide intelligent assistance
"""
import google.genai as genai
from typing import Dict, List

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAcQJtf62wrCEfW9As6XhKJAQXyPcVkre4"
client = genai.Client(api_key=GEMINI_API_KEY)

# System context - describes the application
SYSTEM_CONTEXT = """
You are an AI assistant for the NSE Stock Analysis System. This is a professional stock trading analysis platform with the following features:

## PAGES AND ROUTES:
1. **Dashboard (/)** - Shows latest BUY signals with performance analysis
   - Displays companies with current BUY signals
   - Shows success rates, profit/loss statistics
   - Filters by indicators (SMA, RSI, BB, MACD, STOCH)

2. **Advanced Scanner (/advanced-scanner)** - Historical trading performance scanner
   - Analyzes historical BUY signals
   - Tests target profit % and stop loss % strategies
   - Shows which companies would have been profitable
   - Parameters: target profit %, stop loss %, holding days, date range

3. **Scanner Detail (/scanner-detail/{symbol})** - Detailed company performance
   - Shows individual company trading history
   - Displays all trades with entry/exit prices
   - Shows high/low reached, stop loss analysis
   - Performance by indicator

4. **Symbol Detail (/symbol/{symbol})** - Company stock details
   - Current price and indicators
   - Technical analysis charts
   - Historical data

## TECHNICAL INDICATORS:
- **SMA** (Simple Moving Average): 5, 10, 20, 50, 100, 200 periods
- **RSI** (Relative Strength Index): 7, 14, 21, 50, 80 periods
- **BB** (Bollinger Bands): 10, 20, 50, 100 periods (Lower band)
- **MACD** (Moving Average Convergence Divergence): Short, Long, Standard
- **STOCH** (Stochastic): 5, 9, 14, 21, 50 periods

## KEY CONCEPTS:
- **BUY Signal**: When technical indicator suggests buying
- **Target Profit**: Percentage gain to sell at (e.g., 5%)
- **Stop Loss**: Percentage loss to exit at (e.g., 3%)
- **Success Rate**: Percentage of trades that hit target (includes open trades)
- **Open Trades**: Signals that haven't hit target or stop loss yet
- **Holding Days**: Maximum days to wait for target/stop loss

## DATABASE TABLES:
- daily_prices: Stock price data (open, high, low, close)
- smatbl, rsitbl, bbtbl, macdtbl, stochtbl: Indicator signals
- latest_buy_signals: Current BUY signals

## YOUR ROLE:
- Help users understand the system
- Guide them to the right page/feature
- Explain trading concepts
- Answer questions about indicators
- Provide navigation assistance

When users ask to go somewhere or see something, provide the exact route/URL.
Be professional, concise, and helpful. Use emojis sparingly for clarity.
"""

# Initialize Gemini model
# Using the new google.genai package

# Store conversation history per session
conversation_history: Dict[str, List] = {}


def get_chatbot_response(user_message: str, session_id: str = "default") -> Dict:
    """
    Get response from Gemini AI chatbot
    
    Args:
        user_message: User's question/message
        session_id: Unique session identifier for conversation history
        
    Returns:
        Dict with response and optional redirect URL
    """
    try:
        # Initialize conversation history for new sessions
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Build conversation context
        conversation = conversation_history[session_id]
        
        # Create prompt with system context
        full_prompt = f"{SYSTEM_CONTEXT}\n\n"
        
        # Add conversation history
        for msg in conversation[-5:]:  # Last 5 messages for context
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        
        # Add current user message
        full_prompt += f"User: {user_message}\n"
        full_prompt += "Assistant: "
        
        # Get response from Gemini using correct API format
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        bot_response = response.text
        
        # Store in conversation history
        conversation_history[session_id].append({
            "role": "User",
            "content": user_message
        })
        conversation_history[session_id].append({
            "role": "Assistant",
            "content": bot_response
        })
        
        # Detect if response contains a route/redirect
        redirect_url = detect_redirect(user_message, bot_response)
        
        return {
            "success": True,
            "response": bot_response,
            "redirect": redirect_url
        }
        
    except Exception as e:
        return {
            "success": False,
            "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
            "redirect": None
        }


def detect_redirect(user_message: str, bot_response: str) -> str:
    """
    Detect if user wants to navigate somewhere and return the URL
    """
    user_lower = user_message.lower()
    
    # Dashboard
    if any(word in user_lower for word in ["dashboard", "home", "main page", "latest signals"]):
        return "/"
    
    # Advanced Scanner
    if any(word in user_lower for word in ["scanner", "scan", "historical", "backtest", "test strategy"]):
        return "/advanced-scanner"
    
    # Check if bot response mentions a specific route
    if "/advanced-scanner" in bot_response:
        return "/advanced-scanner"
    elif "/scanner-detail/" in bot_response:
        # Extract symbol if present
        import re
        match = re.search(r'/scanner-detail/([A-Z0-9:]+)', bot_response)
        if match:
            return f"/scanner-detail/{match.group(1)}"
    elif "/symbol/" in bot_response:
        # Extract symbol if present
        import re
        match = re.search(r'/symbol/([A-Z0-9:]+)', bot_response)
        if match:
            return f"/symbol/{match.group(1)}"
    
    return None


def clear_conversation(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in conversation_history:
        conversation_history[session_id] = []
    return {"success": True, "message": "Conversation cleared"}
