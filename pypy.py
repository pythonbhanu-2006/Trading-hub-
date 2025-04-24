
import json
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
import requests
from openai import OpenAI
import os
from flask import Flask, jsonify, request, send_from_directory, session
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)  # Required for session management

# Symbol mapping for yfinance compatibility
SYMBOL_MAPPING = {
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "HG1!": "HG=F", "AL1!": "ALI=F",
    "NI1!": "NICKEL", "ZN1!": "ZINC", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    "CL1!": "CL=F", "BZ1!": "BZ=F", "NG1!": "NG=F", "RB1!": "RB=F",
    "HO1!": "HO=F", "QL1!": "ICI=F", "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X", "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X"
}

# Timeframe limits for yfinance
TIMEFRAME_LIMITS = {
    "1m": timedelta(days=7), "5m": timedelta(days=60), "15m": timedelta(days=60),
    "1h": timedelta(days=730), "4h": timedelta(days=730),
    "1d": None, "1w": None, "1mo": None
}

# Pre-defined market searches
PREDEFINED_MARKETS = [
    "^NSEI: NIFTY 50 (India)", "^GSPC: S&P 500 (USA)", "^DJI: Dow Jones (USA)",
    "^IXIC: NASDAQ (USA)", "^FTSE: FTSE 100 (UK)", "^N225: Nikkei 225 (Japan)",
    "EURUSD=X: EUR/USD", "USDJPY=X: USD/JPY", "GBPUSD=X: GBP/USD", "AUDUSD=X: AUD/USD",
    "XAUUSD: Gold (GC=F)", "XAGUSD: Silver (SI=F)", "CL1!: Crude Oil (CL=F)",
    "BTC-USD: Bitcoin", "ETH-USD: Ethereum", "AAPL: Apple Inc.", "MSFT: Microsoft Corp.",
    "GOOGL: Alphabet Inc.", "AMZN: Amazon.com Inc.", "TSLA: Tesla Inc."
]

# Load symbols from JSON file
def load_symbols(file_path="output.json"):
    try:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Using empty symbol list.")
            return {}
        with open(file_path, "r") as file:
            data = json.load(file)
            symbol_dict = {}
            if "crypto_list" in data:
                symbol_dict.update({code: name for name, code in data["crypto_list"].items()})
            if "indices" in data:
                symbol_dict.update({code: name for code, name in data["indices"].items()})
            if "Metal Commodities" in data:
                symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
                                  for item in data["Metal Commodities"].values()})
            if "Energy Commodities" in data:
                symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
                                  for item in data["Energy Commodities"].values()})
            if "Major Pairs" in data:
                symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
                                  for item in data["Major Pairs"].values()})
            if "Stock" in data and isinstance(data["Stock"], list):
                symbol_dict.update({f"{stock['SYMBOL']}.NS": stock.get("NAME OF COMPANY", stock["SYMBOL"])
                                  for stock in data["Stock"] if "SYMBOL" in stock})
            return symbol_dict
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return {}

# API setup (use environment variables for security)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyDaoOGioXLUJahaWS5vkCcYIs5G4eBqABo")
OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Fetch market data and calculate indicators with TA-Lib
def fetch_market_data(ticker_symbol, start_date, end_date, timeframe="1h"):
    try:
        stock = yf.Ticker(ticker_symbol)
        if start_date:
            historical_data = stock.history(start=start_date, end=end_date, interval=timeframe)
        else:
            historical_data = stock.history(period="max", interval=timeframe)

        if historical_data.empty:
            print(f"No {timeframe} data for {ticker_symbol}. Falling back to 1h timeframe.")
            historical_data = stock.history(start=start_date, end=end_date, interval="1h" if start_date else "max")
            if historical_data.empty:
                print(f"No 1h data either. Falling back to 1d timeframe.")
                historical_data = stock.history(start=start_date, end=end_date, interval="1d" if start_date else "max")
                if historical_data.empty:
                    raise ValueError("No historical data available even with fallback.")

        close_prices = historical_data['Close'].values
        high_prices = historical_data['High'].values
        low_prices = historical_data['Low'].values

        historical_data['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        historical_data['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        historical_data['RSI'] = talib.RSI(close_prices, timeperiod=14)
        historical_data['MACD'], historical_data['MACD_Signal'], _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        historical_data['BB_Upper'], historical_data['BB_Middle'], historical_data['BB_Lower'] = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        historical_data['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        historical_data['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        historical_data['RMI'] = talib.RSI(close_prices, timeperiod=20)

        live_price = stock.info.get('regularMarketPrice', historical_data['Close'].iloc[-1] if not historical_data.empty else None)
        if live_price is None:
            raise ValueError("Unable to fetch live price.")

        return historical_data, live_price
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None, None

# Generate Matplotlib chart in memory
def generate_chart(hist_data, symbol, timeframe):
    plt.figure(figsize=(6, 3))
    data = hist_data.tail(50)  # Last 50 candles for chart
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['SMA_20'], label='SMA 20', color='orange', linestyle='--')
    plt.plot(data.index, data['SMA_50'], label='SMA 50', color='green', linestyle='--')
    plt.title(f'{symbol} {timeframe} Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to memory instead of disk
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return f'data:image/png;base64,{image_base64}'

# Pre-defined analysis style prompts
ANALYSIS_STYLES = {
    "Technical Analysis": """
    You are an expert financial advisor with advanced knowledge of technical analysis. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, including:
    - Buy or Sell signal with entry price
    - Take-Profit (TP) levels (short-term and long-term) adhering to a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level ensuring the specified risk-reward ratio
    - Demand Zone (support) and Supply Zone (resistance) with specific price ranges
    - Confidence level (0-100%) with probabilistic reasoning
    - Detailed rationale using RSI, ADX, MACD, Bollinger Bands, and ATR
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "Trend Following": """
    You are a trend-following expert. Provide a trading recommendation for {symbol} based on {timeframe} candles, emphasizing:
    - Buy or Sell signal with entry
    - Take-Profit (TP) levels aligned with trend, meeting a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level
    - Trend direction, strength, and duration
    - Confidence level (0-100%)
    - Rationale using ADX, SMA crossovers, and MACD
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "Risk-Averse": """
    You are a risk-averse trading specialist. Provide a conservative trading recommendation for {symbol} based on {timeframe} candles, including:
    - Buy or Sell signal with cautious entry
    - Take-Profit (TP) level with modest targets, adhering to a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level with tight risk control
    - Safe entry/exit zones
    - Confidence level (0-100%)
    - Rationale using ATR, RSI, Bollinger Bands
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "SMC (Smart Money Concepts)": """
    You are an expert in Smart Money Concepts. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
    - Buy or Sell signal based on order block or liquidity grab
    - Take-Profit (TP) levels targeting reversal zones, meeting a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level
    - Key order blocks, fair value gaps, and liquidity zones
    - Confidence level (0-100%)
    - Rationale using price structure, ATR, RSI, ADX
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "Price Action": """
    You are a price action trading master. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
    - Buy or Sell signal based on candlestick patterns
    - Take-Profit (TP) levels at key zones, adhering to a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level
    - Support and resistance levels
    - Confidence level (0-100%)
    - Rationale using candlestick formations, price rejection, ATR
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "ICT (Inner Circle Trader)": """
    You are an ICT trading expert. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
    - Buy or Sell signal based on optimal trade entry
    - Take-Profit (TP) levels targeting liquidity pools, meeting a {risk_reward} risk-reward ratio
    - Stop-Loss (SL) level
    - Key levels: breaker blocks, mitigation zones
    - Confidence level (0-100%)
    - Rationale using price structure, ATR, RSI, ADX
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """,
    "Custom Analysis": """
    You are a highly adaptable financial advisor. Provide a trading recommendation for {symbol} based on {timeframe} candles, tailored to the user's custom prompt:
    {custom_prompt}
    Historical Data (all fetched candles): {hist_data}
    Current Live Price: {live_price:.2f}
    Latest Indicators: {indicators}
    Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
    """
}

# Enhanced trading advice from AI
def get_trading_advice(symbol, hist_data, live_price, timeframe, analysis_style, risk_reward, ai_model, custom_prompt=None):
    hist_data_str = hist_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'ADX', 'RMI']].to_string()
    indicators_str = f"""
    - SMA_20: {hist_data['SMA_20'].iloc[-1]:.2f}
    - SMA_50: {hist_data['SMA_50'].iloc[-1]:.2f}
    - RSI: {hist_data['RSI'].iloc[-1]:.2f}
    - MACD: {hist_data['MACD'].iloc[-1]:.2f} (Signal: {hist_data['MACD_Signal'].iloc[-1]:.2f})
    - Bollinger Bands: Upper {hist_data['BB_Upper'].iloc[-1]:.2f}, Lower {hist_data['BB_Lower'].iloc[-1]:.2f}
    - ATR: {hist_data['ATR'].iloc[-1]:.2f}
    - ADX: {hist_data['ADX'].iloc[-1]:.2f}
    - RMI: {hist_data['RMI'].iloc[-1]:.2f}
    """
    if analysis_style == "Custom Analysis" and custom_prompt:
        prompt = ANALYSIS_STYLES[analysis_style].format(
            symbol=symbol,
            timeframe=timeframe,
            hist_data=hist_data_str,
            live_price=live_price,
            indicators=indicators_str,
            custom_prompt=custom_prompt
        )
    else:
        prompt = ANALYSIS_STYLES[analysis_style].format(
            symbol=symbol,
            timeframe=timeframe,
            hist_data=hist_data_str,
            live_price=live_price,
            indicators=indicators_str,
            risk_reward=risk_reward
        )
    system_message = "You are a financial advisor specializing in technical analysis, price action, and institutional trading strategies."

    try:
        if ai_model == "OpenAI":
            response = openai_client.chat.completions.create(
                messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                model="o4-mini",
                temperature=0.7,
                max_tokens=100000
            )
            return response.choices[0].message.content
        else:  # gemini
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [{
                    "parts": [{"text": system_message + "\n\n" + prompt}]
                }]
            }
            url = f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error getting trading advice: {e}"

# Serve index.html at root and /index.html
@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# API Endpoints
@app.route('/api/fetch_advice', methods=['POST'])
def fetch_advice():
    # Track request count
    session['request_count'] = session.get('request_count', 0) + 1
    require_login = session.get('request_count', 0) > 5 and not session.get('logged_in', False)

    if require_login:
        return jsonify({'success': False, 'error': 'Login required', 'requireLogin': True})

    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1h')
        strategy = data.get('strategy', 'intraday')
        risk_reward = data.get('riskReward', '1:3')
        analysis_style = data.get('analysisStyle', 'Technical Analysis')
        ai_model = data.get('aiModel', 'openai')
        custom_prompt = data.get('customPrompt')

        if not symbol:
            return jsonify({'success': False, 'error': 'Please provide a symbol'})

        # Determine date range based on strategy
        today = datetime.now()
        if strategy == 'scalping':
            start_date = today - timedelta(days=7)
        elif strategy == 'intraday':
            start_date = today - timedelta(days=59)
        elif strategy == 'swing':
            start_date = today - timedelta(days=730)
        else:  # long_term
            start_date = None

        # Adjust for timeframe limits
        max_delta = TIMEFRAME_LIMITS.get(timeframe)
        warning = ""
        if max_delta and start_date and (today - start_date) > max_delta:
            start_date = today - max_delta
            warning = f"Warning: {timeframe} data limited to {max_delta.days} days. Adjusted start date to {start_date.strftime('%Y-%m-%d')}.\n"

        end_date = today
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d')

        # Fetch market data
        yf_symbol = SYMBOL_MAPPING.get(symbol, symbol)
        hist_data, live_price = fetch_market_data(yf_symbol, start_str, end_str, timeframe)

        if hist_data is None or live_price is None:
            return jsonify({'success': False, 'error': f'Failed to fetch data for {symbol} ({yf_symbol})'})

        # Get trading advice with all historical data
        if analysis_style == "Custom Analysis" and not custom_prompt:
            return jsonify({'success': False, 'error': 'Please provide a custom prompt for Custom Analysis'})

        advice = get_trading_advice(
            symbol, hist_data, live_price, timeframe, analysis_style, risk_reward, ai_model, custom_prompt
        )

        # Generate chart (in memory)
        chart_data = generate_chart(hist_data, symbol, timeframe)

        output = (
            f"{warning}"
            f"Last Traded Price for {symbol} ({yf_symbol}): {live_price:.2f}\n"
            f"\nFetching trading advice based on {timeframe} candles from {start_str or 'max'} to {end_str} "
            f"(Strategy: {strategy}, Style: {analysis_style}, Risk-Reward: {risk_reward}, AI Model: {ai_model})...\n"
            f"\nTrading Advice:\n{advice}"
        )

        return jsonify({
            'success': True,
            'output': output,
            'chartData': chart_data,
            'requestCount': session.get('request_count', 0)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    try:
        symbol_data = load_symbols()
        symbols_list = PREDEFINED_MARKETS + [f'{code}: {name}' for code, name in symbol_data.items()]
        return jsonify({'success': True, 'symbols': symbols_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # Simple authentication for demonstration (replace with proper auth system)
        if username == "user" and password == "password":
            session['logged_in'] = True
            session['request_count'] = 0  # Reset request count after login
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response to suppress favicon errors

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# import json
# import yfinance as yf
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np
# import talib
# import requests
# from openai import OpenAI
# import os
# from flask import Flask, jsonify, request, send_from_directory
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend for server-side plotting
# import matplotlib.pyplot as plt
# import uuid

# app = Flask(__name__, static_folder='static')

# # Create charts directory
# CHARTS_DIR = os.path.join('static', 'charts')
# os.makedirs(CHARTS_DIR, exist_ok=True)

# # Symbol mapping for yfinance compatibility
# SYMBOL_MAPPING = {
#     "XAUUSD": "GC=F", "XAGUSD": "SI=F", "HG1!": "HG=F", "AL1!": "ALI=F",
#     "NI1!": "NICKEL", "ZN1!": "ZINC", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
#     "CL1!": "CL=F", "BZ1!": "BZ=F", "NG1!": "NG=F", "RB1!": "RB=F",
#     "HO1!": "HO=F", "QL1!": "ICI=F", "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
#     "GBPUSD": "GBPUSD=X", "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X",
#     "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X"
# }

# # Timeframe limits for yfinance
# TIMEFRAME_LIMITS = {
#     "1m": timedelta(days=7), "5m": timedelta(days=60), "15m": timedelta(days=60),
#     "1h": timedelta(days=730), "4h": timedelta(days=730),
#     "1d": None, "1w": None, "1mo": None
# }

# # Pre-defined market searches
# PREDEFINED_MARKETS = [
#     "^NSEI: NIFTY 50 (India)", "^GSPC: S&P 500 (USA)", "^DJI: Dow Jones (USA)",
#     "^IXIC: NASDAQ (USA)", "^FTSE: FTSE 100 (UK)", "^N225: Nikkei 225 (Japan)",
#     "EURUSD=X: EUR/USD", "USDJPY=X: USD/JPY", "GBPUSD=X: GBP/USD", "AUDUSD=X: AUD/USD",
#     "XAUUSD: Gold (GC=F)", "XAGUSD: Silver (SI=F)", "CL1!: Crude Oil (CL=F)",
#     "BTC-USD: Bitcoin", "ETH-USD: Ethereum", "AAPL: Apple Inc.", "MSFT: Microsoft Corp.",
#     "GOOGL: Alphabet Inc.", "AMZN: Amazon.com Inc.", "TSLA: Tesla Inc."
# ]

# # Load symbols from JSON file
# def load_symbols(file_path="output.json"):
#     try:
#         if not os.path.exists(file_path):
#             print(f"Warning: {file_path} not found. Using empty symbol list.")
#             return {}
#         with open(file_path, "r") as file:
#             data = json.load(file)
#             symbol_dict = {}
#             if "crypto_list" in data:
#                 symbol_dict.update({code: name for name, code in data["crypto_list"].items()})
#             if "indices" in data:
#                 symbol_dict.update({code: name for code, name in data["indices"].items()})
#             if "Metal Commodities" in data:
#                 symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
#                                   for item in data["Metal Commodities"].values()})
#             if "Energy Commodities" in data:
#                 symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
#                                   for item in data["Energy Commodities"].values()})
#             if "Major Pairs" in data:
#                 symbol_dict.update({item["symbol"]: item.get("name", item["symbol"])
#                                   for item in data["Major Pairs"].values()})
#             if "Stock" in data and isinstance(data["Stock"], list):
#                 symbol_dict.update({f"{stock['SYMBOL']}.NS": stock.get("NAME OF COMPANY", stock["SYMBOL"])
#                                   for stock in data["Stock"] if "SYMBOL" in stock})
#             return symbol_dict
#     except Exception as e:
#         print(f"Error loading symbols: {e}")
#         return {}

# # API setup (use environment variables for security)
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
# OPENAI_BASE_URL = "https://models.inference.ai.azure.com"
# GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# # OpenAI client
# openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# # Fetch market data and calculate indicators with TA-Lib
# def fetch_market_data(ticker_symbol, start_date, end_date, timeframe="1h"):
#     try:
#         stock = yf.Ticker(ticker_symbol)
#         if start_date:
#             historical_data = stock.history(start=start_date, end=end_date, interval=timeframe)
#         else:
#             historical_data = stock.history(period="max", interval=timeframe)

#         if historical_data.empty:
#             print(f"No {timeframe} data for {ticker_symbol}. Falling back to 1h timeframe.")
#             historical_data = stock.history(start=start_date, end=end_date, interval="1h" if start_date else "max")
#             if historical_data.empty:
#                 print(f"No 1h data either. Falling back to 1d timeframe.")
#                 historical_data = stock.history(start=start_date, end=end_date, interval="1d" if start_date else "max")
#                 if historical_data.empty:
#                     raise ValueError("No historical data available even with fallback.")

#         close_prices = historical_data['Close'].values
#         high_prices = historical_data['High'].values
#         low_prices = historical_data['Low'].values

#         historical_data['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
#         historical_data['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
#         historical_data['RSI'] = talib.RSI(close_prices, timeperiod=14)
#         historical_data['MACD'], historical_data['MACD_Signal'], _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
#         historical_data['BB_Upper'], historical_data['BB_Middle'], historical_data['BB_Lower'] = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
#         historical_data['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
#         historical_data['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
#         historical_data['RMI'] = talib.RSI(close_prices, timeperiod=20)

#         live_price = stock.info.get('regularMarketPrice', historical_data['Close'].iloc[-1] if not historical_data.empty else None)
#         if live_price is None:
#             raise ValueError("Unable to fetch live price.")

#         return historical_data, live_price
#     except Exception as e:
#         print(f"Error fetching data for {ticker_symbol}: {e}")
#         return None, None

# # Generate Matplotlib chart
# def generate_chart(hist_data, symbol, timeframe):
#     plt.figure(figsize=(6, 3))
#     data = hist_data.tail(50)  # Last 50 candles for chart
#     plt.plot(data.index, data['Close'], label='Close Price', color='blue')
#     plt.plot(data.index, data['SMA_20'], label='SMA 20', color='orange', linestyle='--')
#     plt.plot(data.index, data['SMA_50'], label='SMA 50', color='green', linestyle='--')
#     plt.title(f'{symbol} {timeframe} Chart')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     chart_id = str(uuid.uuid4())
#     chart_path = os.path.join(CHARTS_DIR, f'chart_{chart_id}.png')
#     plt.savefig(chart_path, format='png')
#     plt.close()
#     return f'/static/charts/chart_{chart_id}.png'

# # Pre-defined analysis style prompts
# ANALYSIS_STYLES = {
#     "Technical Analysis": """
#     You are an expert financial advisor with advanced knowledge of technical analysis. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, including:
#     - Buy or Sell signal with entry price
#     - Take-Profit (TP) levels (short-term and long-term) adhering to a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level ensuring the specified risk-reward ratio
#     - Demand Zone (support) and Supply Zone (resistance) with specific price ranges
#     - Confidence level (0-100%) with probabilistic reasoning
#     - Detailed rationale using RSI, ADX, MACD, Bollinger Bands, and ATR
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "Trend Following": """
#     You are a trend-following expert. Provide a trading recommendation for {symbol} based on {timeframe} candles, emphasizing:
#     - Buy or Sell signal with entry
#     - Take-Profit (TP) levels aligned with trend, meeting a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level
#     - Trend direction, strength, and duration
#     - Confidence level (0-100%)
#     - Rationale using ADX, SMA crossovers, and MACD
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "Risk-Averse": """
#     You are a risk-averse trading specialist. Provide a conservative trading recommendation for {symbol} based on {timeframe} candles, including:
#     - Buy or Sell signal with cautious entry
#     - Take-Profit (TP) level with modest targets, adhering to a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level with tight risk control
#     - Safe entry/exit zones
#     - Confidence level (0-100%)
#     - Rationale using ATR, RSI, Bollinger Bands
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "SMC (Smart Money Concepts)": """
#     You are an expert in Smart Money Concepts. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
#     - Buy or Sell signal based on order block or liquidity grab
#     - Take-Profit (TP) levels targeting reversal zones, meeting a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level
#     - Key order blocks, fair value gaps, and liquidity zones
#     - Confidence level (0-100%)
#     - Rationale using price structure, ATR, RSI, ADX
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "Price Action": """
#     You are a price action trading master. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
#     - Buy or Sell signal based on candlestick patterns
#     - Take-Profit (TP) levels at key zones, adhering to a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level
#     - Support and resistance levels
#     - Confidence level (0-100%)
#     - Rationale using candlestick formations, price rejection, ATR
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "ICT (Inner Circle Trader)": """
#     You are an ICT trading expert. Provide a trading recommendation for {symbol} based on {timeframe} candles, including:
#     - Buy or Sell signal based on optimal trade entry
#     - Take-Profit (TP) levels targeting liquidity pools, meeting a {risk_reward} risk-reward ratio
#     - Stop-Loss (SL) level
#     - Key levels: breaker blocks, mitigation zones
#     - Confidence level (0-100%)
#     - Rationale using price structure, ATR, RSI, ADX
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """,
#     "Custom Analysis": """
#     You are a highly adaptable financial advisor. Provide a trading recommendation for {symbol} based on {timeframe} candles, tailored to the user's custom prompt:
#     {custom_prompt}
#     Historical Data (all fetched candles): {hist_data}
#     Current Live Price: {live_price:.2f}
#     Latest Indicators: {indicators}
#     Note: Analyze the full historical data provided, focusing on recent trends and key patterns relevant to the timeframe.
#     """
# }

# # Enhanced trading advice from AI
# def get_trading_advice(symbol, hist_data, live_price, timeframe, analysis_style, risk_reward, ai_model, custom_prompt=None):
#     # Send all historical data instead of last 10 candles
#     hist_data_str = hist_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'ADX', 'RMI']].to_string()
#     indicators_str = f"""
#     - SMA_20: {hist_data['SMA_20'].iloc[-1]:.2f}
#     - SMA_50: {hist_data['SMA_50'].iloc[-1]:.2f}
#     - RSI: {hist_data['RSI'].iloc[-1]:.2f}
#     - MACD: {hist_data['MACD'].iloc[-1]:.2f} (Signal: {hist_data['MACD_Signal'].iloc[-1]:.2f})
#     - Bollinger Bands: Upper {hist_data['BB_Upper'].iloc[-1]:.2f}, Lower {hist_data['BB_Lower'].iloc[-1]:.2f}
#     - ATR: {hist_data['ATR'].iloc[-1]:.2f}
#     - ADX: {hist_data['ADX'].iloc[-1]:.2f}
#     - RMI: {hist_data['RMI'].iloc[-1]:.2f}
#     """
#     if analysis_style == "Custom Analysis" and custom_prompt:
#         prompt = ANALYSIS_STYLES[analysis_style].format(
#             symbol=symbol,
#             timeframe=timeframe,
#             hist_data=hist_data_str,
#             live_price=live_price,
#             indicators=indicators_str,
#             custom_prompt=custom_prompt
#         )
#     else:
#         prompt = ANALYSIS_STYLES[analysis_style].format(
#             symbol=symbol,
#             timeframe=timeframe,
#             hist_data=hist_data_str,
#             live_price=live_price,
#             indicators=indicators_str,
#             risk_reward=risk_reward
#         )
#     system_message = "You are a financial advisor specializing in technical analysis, price action, and institutional trading strategies."

#     try:
#         if ai_model == "openai":
#             response = openai_client.chat.completions.create(
#                 messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
#                 model="gpt-4o",
#                 temperature=0.7,
#                 max_tokens=4096
#             )
#             return response.choices[0].message.content
#         else:  # gemini
#             headers = {
#                 "Content-Type": "application/json"
#             }
#             data = {
#                 "contents": [{
#                     "parts": [{"text": system_message + "\n\n" + prompt}]
#                 }]
#             }
#             url = f"{GEMINI_BASE_URL}?key={GEMINI_API_KEY}"
#             response = requests.post(url, headers=headers, json=data)
#             response.raise_for_status()
#             result = response.json()
#             return result["candidates"][0]["content"]["parts"][0]["text"]
#     except Exception as e:
#         return f"Error getting trading advice: {e}"

# # Serve index.html at root and /index.html
# @app.route('/')
# @app.route('/index.html')
# def serve_index():
#     return send_from_directory(app.static_folder, 'index.html')

# # Serve static files (charts)
# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory(app.static_folder, filename)

# # API Endpoints
# @app.route('/api/fetch_advice', methods=['POST'])
# def fetch_advice():
#     try:
#         data = request.get_json()
#         symbol = data.get('symbol')
#         timeframe = data.get('timeframe', '1h')
#         strategy = data.get('strategy', 'intraday')
#         risk_reward = data.get('riskReward', '1:3')
#         analysis_style = data.get('analysisStyle', 'Technical Analysis')
#         ai_model = data.get('aiModel', 'openai')
#         custom_prompt = data.get('customPrompt')

#         if not symbol:
#             return jsonify({'success': False, 'error': 'Please provide a symbol'})

#         # Determine date range based on strategy
#         today = datetime.now()
#         if strategy == 'scalping':
#             start_date = today - timedelta(days=7)
#         elif strategy == 'intraday':
#             start_date = today - timedelta(days=59)
#         elif strategy == 'swing':
#             start_date = today - timedelta(days=730)
#         else:  # long_term
#             start_date = None

#         # Adjust for timeframe limits
#         max_delta = TIMEFRAME_LIMITS.get(timeframe)
#         warning = ""
#         if max_delta and start_date and (today - start_date) > max_delta:
#             start_date = today - max_delta
#             warning = f"Warning: {timeframe} data limited to {max_delta.days} days. Adjusted start date to {start_date.strftime('%Y-%m-%d')}.\n"

#         end_date = today
#         start_str = start_date.strftime('%Y-%m-%d') if start_date else None
#         end_str = end_date.strftime('%Y-%m-%d')

#         # Fetch market data
#         yf_symbol = SYMBOL_MAPPING.get(symbol, symbol)
#         hist_data, live_price = fetch_market_data(yf_symbol, start_str, end_str, timeframe)

#         if hist_data is None or live_price is None:
#             return jsonify({'success': False, 'error': f'Failed to fetch data for {symbol} ({yf_symbol})'})

#         # Get trading advice with all historical data
#         if analysis_style == "Custom Analysis" and not custom_prompt:
#             return jsonify({'success': False, 'error': 'Please provide a custom prompt for Custom Analysis'})

#         advice = get_trading_advice(
#             symbol, hist_data, live_price, timeframe, analysis_style, risk_reward, ai_model, custom_prompt
#         )

#         # Generate chart (last 50 candles)
#         chart_url = generate_chart(hist_data, symbol, timeframe)

#         output = (
#             f"{warning}"
#             f"Last Traded Price for {symbol} ({yf_symbol}): {live_price:.2f}\n"
#             f"\nFetching trading advice based on {timeframe} candles from {start_str or 'max'} to {end_str} "
#             f"(Strategy: {strategy}, Style: {analysis_style}, Risk-Reward: {risk_reward}, AI Model: {ai_model})...\n"
#             f"\nTrading Advice:\n{advice}"
#         )

#         return jsonify({
#             'success': True,
#             'output': output,
#             'chartUrl': chart_url
#         })

#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/api/symbols', methods=['GET'])
# def get_symbols():
#     try:
#         symbol_data = load_symbols()
#         symbols_list = PREDEFINED_MARKETS + [f'{code}: {name}' for code, name in symbol_data.items()]
#         return jsonify({'success': True, 'symbols': symbols_list})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/favicon.ico')
# def favicon():
#     return '', 204  # No content response to suppress favicon errors

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
