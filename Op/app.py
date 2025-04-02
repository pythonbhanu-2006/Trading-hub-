import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import io
import base64
import re
from datetime import datetime, timedelta
import yfinance as yf
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patheffects
from PIL import Image, ImageDraw, ImageFont

# Set page config
st.set_page_config(
    page_title="Financial Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for dark mode and clean design
st.markdown("""
<style>
    /* Dark theme */
    body {
        background-color: #1E1E1E;
        color: white;
    }
    
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Style for title */
    .title {
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Style for labels */
    .label {
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    
    /* Style for dropdown */
    .stSelectbox > div > div {
        background-color: #333333;
        color: white;
        border: 1px solid #444444;
    }
    
    /* Style for text fields */
    .stTextInput > div > div > input {
        background-color: #333333;
        color: white;
        border: 1px solid #444444;
    }
    
    /* Style for textarea */
    .stTextArea > div > div > textarea {
        background-color: #333333;
        color: white;
        border: 1px solid #444444;
    }
    
    /* Style for button */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    /* Quick access category styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #333333;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    }
    
    /* Adjust column spacing and alignment */
    .row-widget.stHorizontal {
        gap: 10px;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Symbol mapping for yfinance compatibility
SYMBOL_MAPPING = {
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "HG1!": "HG=F", "AL1!": "ALI=F",
    "NI1!": "NICKEL", "ZN1!": "ZINC", "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    "CL1!": "CL=F", "BZ1!": "BZ=F", "NG1!": "NG=F", "RB1!": "RB=F",
    "HO1!": "HO=F", "QL1!": "ICI=F", "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X", "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X",
    # Fix for Indian stocks
    "TATAMOTORS": "TATAMOTORS.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS", "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS",
    "AXISBANK": "AXISBANK.NS", "SBIN": "SBIN.NS", "WIPRO": "WIPRO.NS",
    "SUNPHARMA": "SUNPHARMA.NS", "BAJFINANCE": "BAJFINANCE.NS"
}

# Timeframe options
TIMEFRAMES = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m", 
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d",
    "1 Week": "1wk"
}

# Strategy options with their allowed timeframes and data limits
STRATEGIES = {
    "Scalping (7 days)": {"value": "scalping", "timeframes": ["1 Minute"], "days": 7},
    "Intraday (60 days)": {"value": "intraday", "timeframes": ["5 Minutes", "15 Minutes"], "days": 60},
    "Swing Trading (730 days)": {"value": "swing", "timeframes": ["1 Hour", "4 Hours"], "days": 730},
    "Long Term (max)": {"value": "longterm", "timeframes": ["1 Day", "1 Week"], "days": 2000}
}

# Risk-Reward Ratios
RISK_REWARD_RATIOS = ["1:1", "1:2", "1:3", "1:4", "1:5", "2:1", "3:1"]

# Analysis styles
ANALYSIS_STYLES = {
    "Technical Analysis": """
        You are an expert financial advisor with advanced knowledge of technical analysis. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, including:
        - Buy or Sell signal with entry price
        - Take-Profit (TP) levels (short-term and long-term) adhering to a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level ensuring the specified risk-reward ratio
        - Demand Zone (support) and Supply Zone (resistance) with specific price ranges
        - Confidence level (0-100%) with probabilistic reasoning
        Current Live Price: {live_price:.2f}
        
        Use these advanced technical indicators in your analysis:
        1. Moving Averages: SMA 5, 10, 20, 50, 100, 200 and EMA 5, 10, 20, 50, 100, 200
        2. MACD Line, Signal Line, and Histogram
        3. RSI (14)
        4. Stochastic Oscillator (K and D)
        5. ATR (14) - Volatility indicator
        6. Bollinger Bands (Upper, Middle, Lower, Width)
        7. ADX (Trend Strength) with DI+ and DI-
        8. On-Balance Volume (OBV) for volume confirmation
        9. VWAP (Volume-Weighted Average Price)
        10. Williams %R
        11. CCI (Commodity Channel Index)
        12. Parabolic SAR
        13. Ichimoku Cloud Components (Tenkan, Kijun, Senkou A/B, Chikou)
        14. Keltner Channels
        15. Rate of Change (ROC)
        16. Money Flow Index (MFI)
        17. Chaikin Money Flow (CMF)
        18. Fibonacci Retracement Levels
        19. Force Index
        20. Ease of Movement
        21. Coppock Curve
        22. Aroon Indicator (Up, Down, Oscillator)
        23. Relative Vigor Index (RVI)
        24. Detrended Price Oscillator
        25. Mass Index
        26. Know Sure Thing (KST)
        27. Percentage Price Oscillator (PPO)
        28. Awesome Oscillator
        29. Ultimate Oscillator
        30. Hull Moving Average
        31. SuperTrend indicator
        32. Composite Score (0-100 aggregated technical strength)
    """,
    "Trend Following": """
        You are a trend-following expert with a deep understanding of momentum trading. Provide an in-depth trading recommendation for {symbol} based on {timeframe} candles, emphasizing:
        - Buy or Sell signal with precise entry based on trend confirmation
        - Take-Profit (TP) levels aligned with trend continuation, meeting a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level to protect against reversals, maintaining the risk-reward ratio
        - Trend direction, strength, and potential duration
        - Confidence level (0-100%) with trend persistence probability
        Current Live Price: {live_price:.2f}
    """,
    "Support/Resistance": """
        You are an expert in price structure analysis. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, including:
        - Buy or Sell signal with entry based on key levels
        - Take-Profit (TP) targets at the next significant resistance/support levels, with a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level beyond the nearest support/resistance, maintaining the risk-reward ratio
        - Key support and resistance levels with price ranges
        - Confidence level (0-100%) with reasoning
        Current Live Price: {live_price:.2f}
    """,
    "Pattern Recognition": """
        You are a chart pattern specialist. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, including:
        - Buy or Sell signal based on chart pattern identification
        - Take-Profit (TP) levels projected from pattern measurement, adhering to a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level at pattern invalidation point, ensuring the risk-reward ratio
        - Identified patterns (head & shoulders, triangles, flags, etc.) with completion percentage
        - Confidence level (0-100%) with pattern reliability assessment
        Current Live Price: {live_price:.2f}
    """,
    "Price Action": """
        You are a price action trading master, relying on raw market movements. Provide an in-depth trading recommendation for {symbol} based on {timeframe} candles, including:
        - Buy or Sell signal based on candlestick patterns or structural shifts
        - Take-Profit (TP) levels at key rejection or breakout zones, adhering to a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level beyond recent swing points, ensuring the risk-reward ratio
        - Support and resistance levels with price rejection evidence
        - Confidence level (0-100%) with pattern reliability analysis
        Current Live Price: {live_price:.2f}
    """,
    "Elliott Wave Theory": """
        You are an Elliott Wave Theory expert. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Current wave count and position within the larger wave structure
        - Buy or Sell signal based on Elliott Wave principles and wave termination points
        - Take-Profit (TP) levels projected according to Fibonacci extensions of waves, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level based on wave invalidation points
        - Confidence level (0-100%) based on wave pattern clarity
        - Wave targets for the next expected market movement
        Current Live Price: {live_price:.2f}
    """,
    "Wyckoff Method": """
        You are a Wyckoff Method specialist. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, using Wyckoff principles:
        - Identify current Wyckoff Phase (Accumulation, Markup, Distribution, Markdown)
        - Buy or Sell signal with entry based on Wyckoff schematics (springs, tests, signs of strength/weakness)
        - Take-Profit (TP) levels based on projected price targets following Wyckoff's cause and effect principle, with {risk_reward} risk-reward ratio
        - Stop-Loss (SL) level based on Wyckoff invalidation points
        - Volume analysis and its confirmation/divergence from price action
        - Confidence level (0-100%) with Wyckoff schematic compliance assessment
        Current Live Price: {live_price:.2f}
    """,
    "Market Profile": """
        You are a Market Profile & Volume Profile trading expert. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Identification of Value Area (VAH, VAL, POC)
        - Buy or Sell signal based on price's position relative to value areas and acceptance/rejection
        - Entry points near high-volume nodes or during breakouts from low-volume nodes
        - Take-Profit (TP) levels targeted at the next significant volume node, adhering to {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond relevant low-volume areas
        - Balance vs. imbalance areas and their trading implications
        - Confidence level (0-100%) with reasoning
        Current Live Price: {live_price:.2f}
    """,
    "Order Flow Analysis": """
        You are an Order Flow and Market Microstructure expert. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Analysis of buying/selling pressure based on order flow patterns
        - Buy or Sell signal based on order flow imbalances and liquidity voids
        - Entry points around stop hunts, liquidation levels, or institutional support/resistance
        - Take-Profit (TP) targets at major liquidity pools, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond key market structure points
        - Smart money movement patterns and their implications
        - Confidence level (0-100%) with explanation of order flow conditions
        Current Live Price: {live_price:.2f}
    """,
    "Harmonic Patterns": """
        You are a Harmonic Pattern trading specialist. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Identification of active harmonic patterns (Gartley, Butterfly, Bat, Crab, Shark, Cypher, etc.)
        - Pattern completion percentage and quality assessment
        - Buy or Sell signal with precise entry points at pattern completion
        - Take-Profit (TP) levels using pattern extension targets, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement based on pattern invalidation rules
        - Fibonacci confluence with pattern completion points
        - Confidence level (0-100%) based on pattern clarity and precision
        Current Live Price: {live_price:.2f}
    """,
    "Intermarket Analysis": """
        You are an Intermarket Analysis expert. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles while considering:
        - Correlations and divergences with related markets (currencies, commodities, bonds, indices)
        - Buy or Sell signal based on intermarket relationships and relative strength/weakness
        - Take-Profit (TP) levels based on key intermarket pivot points, with a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement considering correlated asset movements
        - Analysis of sector rotation and money flow between markets
        - Leading indicator signals from related markets
        - Confidence level (0-100%) with intermarket confirmation assessment
        Current Live Price: {live_price:.2f}
    """,
    "Supply & Demand Zones": """
        You are a Supply and Demand trading expert. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Identification of institutional Supply and Demand zones
        - Zone quality assessment based on formation criteria (fresh zones, strong momentum, quick departures)
        - Buy or Sell signal with entry at zone retest with confluence
        - Take-Profit (TP) targets at opposing zones, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond zone boundaries
        - Zone strength classification (weak/moderate/strong)
        - Confidence level (0-100%) based on zone quality and market context
        Current Live Price: {live_price:.2f}
    """,
    "Auction Market Theory": """
        You are an Auction Market Theory expert. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, using Auction Market principles:
        - Analysis of the current auction process (discovery, facilitation, termination)
        - Buy or Sell signal based on auction inefficiencies and excess
        - Entry points near failed auctions, excess, or single prints
        - Take-Profit (TP) levels at opposing auction extremes, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement at points of auction invalidation
        - Fair value gaps identification and their trading implications
        - Confidence level (0-100%) with auction theory compliance assessment
        Current Live Price: {live_price:.2f}
    """,
    "Market Cycles Analysis": """
        You are a Market Cycles specialist. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Current position within market cycles (accumulation, markup, distribution, markdown)
        - Cycle phase timing and duration analysis
        - Buy or Sell signal based on cycle positioning and transition points
        - Take-Profit (TP) targets aligned with cycle progression, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement considering cycle invalidation points
        - Seasonality factors and their influence on current cycle
        - Confidence level (0-100%) with cycle recognition reliability assessment
        Current Live Price: {live_price:.2f}
    """,
    "VWAP Trading": """
        You are a Volume-Weighted Average Price (VWAP) trading expert. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Relationship between price and standard/anchored VWAP levels
        - VWAP bands (1, 2, 3 standard deviations) analysis
        - Buy or Sell signal based on VWAP rejections, crosses, or band tests
        - Entry points at key VWAP-based decision zones
        - Take-Profit (TP) levels at opposing VWAP bands, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond relevant VWAP levels
        - Volume profile confirmation of VWAP signals
        - Confidence level (0-100%) with VWAP signal strength assessment
        Current Live Price: {live_price:.2f}
    """,
    "Smart Money Concept (SMC)": """
        You are a Smart Money Concept (SMC) trading expert. Provide a comprehensive trading recommendation for {symbol} based on {timeframe} candles, focusing on:
        - Identification of Smart Money movements (institutional manipulation patterns)
        - Liquidity grabs/sweeps (stop hunts) above/below key swing points
        - Order blocks, breaker blocks, and fair value gaps
        - Mitigation points and optimal entry zones
        - Premium/discount zones and their trading implications
        - Buy or Sell signal with precise entries at institutional order blocks
        - Take-Profit (TP) targets at opposing order blocks/imbalance points, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond valid swing points
        - Market structure shifts and their implications for trend continuation/reversal
        - Confidence level (0-100%) with full SMC analysis rationale
        Current Live Price: {live_price:.2f}
    """,
    "ICT Concepts": """
        You are an Inner Circle Trader (ICT) methodology expert. Provide a detailed trading recommendation for {symbol} based on {timeframe} candles, using ICT concepts:
        - Market structure (higher highs/higher lows or lower highs/lower lows)
        - Optimal Trade Entry (OTE) identification 
        - Fair Value Gaps (FVGs) and their importance
        - Order Blocks (OBs) and Breaker Blocks (BBs)
        - Liquidity engineering and stop hunts
        - Buy or Sell signal with entry at key ICT concepts (FVG, OB, OTE)
        - Take-Profit (TP) targets at opposing liquidity voids, maintaining a {risk_reward} risk-reward ratio
        - Stop-Loss (SL) placement beyond key swing points
        - Money flow and smart money funding/distribution patterns
        - Confidence level (0-100%) with full ICT methodology assessment
        Current Live Price: {live_price:.2f}
    """,
    "Custom Analysis": """
        {custom_prompt}
        
        Current Symbol: {symbol}
        Current Timeframe: {timeframe}
        Risk-Reward Ratio: {risk_reward}
        Current Live Price: {live_price:.2f}
    """
}

# Load symbols from JSON
@st.cache_data
def load_symbols():
    try:
        # First try to load from local file
        try:
            with open('attached_assets/output.json', 'r') as f:
                data = json.load(f)
        except:
            try:
                with open('data/symbols.json', 'r') as f:
                    data = json.load(f)
            except:
                # If local file not found, try to fetch from URL
                url = "https://raw.githubusercontent.com/pythonbhanu-2006/Bhanu-/main/output.json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                else:
                    raise Exception(f"Failed to fetch JSON: {response.status_code}")
        
        symbol_dict = {}
        asset_counts = {
            "Cryptocurrencies": 0,
            "Indices": 0,
            "Metal Commodities": 0,
            "Energy Commodities": 0,
            "Forex Pairs": 0,
            "Indian Stocks": 0,
            "Other": 0
        }
        
        # Process different sections of the JSON
        if "crypto_list" in data:
            for name, code in data["crypto_list"].items():
                symbol_dict[code] = name
                asset_counts["Cryptocurrencies"] += 1
                
        if "indices" in data:
            for code, name in data["indices"].items():
                symbol_dict[code] = name
                asset_counts["Indices"] += 1
                
        if "Metal Commodities" in data:
            for name, details in data["Metal Commodities"].items():
                symbol = details.get("symbol", "")
                if symbol:
                    # Map to Yahoo Finance compatible symbols
                    if symbol in SYMBOL_MAPPING:
                        symbol = SYMBOL_MAPPING[symbol]
                    symbol_dict[symbol] = name
                    asset_counts["Metal Commodities"] += 1
                    
        if "Energy Commodities" in data:
            for name, details in data["Energy Commodities"].items():
                symbol = details.get("symbol", "")
                if symbol:
                    # Map to Yahoo Finance compatible symbols
                    if symbol in SYMBOL_MAPPING:
                        symbol = SYMBOL_MAPPING[symbol]
                    symbol_dict[symbol] = name
                    asset_counts["Energy Commodities"] += 1
                    
        if "Major Pairs" in data:
            for name, details in data["Major Pairs"].items():
                symbol = details.get("symbol", "")
                if symbol:
                    # Map to Yahoo Finance compatible symbols
                    if symbol in SYMBOL_MAPPING:
                        symbol = SYMBOL_MAPPING[symbol]
                    else:
                        # Add =X suffix for forex symbols if not already there
                        if not symbol.endswith('=X'):
                            symbol = f"{symbol}=X"
                    symbol_dict[symbol] = name
                    asset_counts["Forex Pairs"] += 1
        
        # Process Indian stocks
        if "Stock" in data and isinstance(data["Stock"], list):
            for stock in data["Stock"]:
                if "SYMBOL" in stock and "NAME OF COMPANY" in stock:
                    # Add .NS suffix for NSE stocks
                    symbol = f"{stock['SYMBOL']}.NS"
                    symbol_dict[symbol] = stock['NAME OF COMPANY']
                    asset_counts["Indian Stocks"] += 1
        
        # Add predefined defaults if not in the loaded list
        key_symbols = {
            "^NSEI": "NIFTY 50 (India)",
            "^BSESN": "SENSEX (India)",
            "^NSEBANK": "Bank NIFTY (India)",
            "BTC-USD": "Bitcoin",
            "EURUSD=X": "EUR/USD"
        }
        for sym, name in key_symbols.items():
            if sym not in symbol_dict:
                symbol_dict[sym] = name
                asset_counts["Other"] += 1
        
        # Store asset counts in session state
        st.session_state.asset_counts = asset_counts
        st.session_state.total_assets = sum(asset_counts.values())
                
        return symbol_dict
            
    except Exception as e:
        print(f"Error loading symbols: {e}")
        # Default fallback if loading fails
        return {
            "^NSEI": "NIFTY 50 (India)",
            "^BSESN": "SENSEX (India)",
            "^NSEBANK": "Bank NIFTY (India)",
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "EURUSD=X": "EUR/USD",
            "USDJPY=X": "USD/JPY",
            "GC=F": "Gold",
            "CL=F": "Crude Oil"
        }

# Fetch market data from Yahoo Finance
def fetch_market_data(symbol, timeframe="1d", days=60):
    try:
        # Map symbol if needed
        if symbol in SYMBOL_MAPPING:
            yf_symbol = SYMBOL_MAPPING[symbol]
        else:
            # Check if it's Indian stock but without .NS suffix
            if symbol.upper() in ["TATAMOTORS", "TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", 
                                "AXISBANK", "SBIN", "WIPRO", "SUNPHARMA", "BAJFINANCE"]:
                yf_symbol = f"{symbol.upper()}.NS"
                print(f"Mapped Indian stock {symbol} to {yf_symbol}")
            else:
                yf_symbol = symbol
            
        # Set period based on timeframe and days
        # For Scalping (1m) use 7d (Yahoo Finance limit)
        # For Intraday (5m, 15m) use 60d (Yahoo Finance limit)
        # For Swing (1h, 4h) use 730d (2 years)
        # For Long Term (1d, 1w) use "max" instead of days
        if timeframe.lower() in ["1m"]:
            period = "7d"  # Yahoo limits for 1m
        elif timeframe.lower() in ["5m", "15m"]:
            period = "60d"  # Yahoo limits for 5m and 15m
        elif timeframe.lower() in ["1h", "4h"]:
            period = "730d" # 2 years for hourly data
        elif timeframe.lower() in ["1d", "1wk"]:
            period = "max"  # Max available for daily and weekly
        else:
            # Fallback to provided days or default
            period = f"{days}d"
            
        # Convert user-friendly timeframe to yfinance format
        interval = timeframe.lower()
        
        # For debugging
        print(f"Fetching {yf_symbol} data with interval={interval}, period={period}")
        
        # Get data
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            # Try fallback to daily data
            print(f"No {interval} data for {yf_symbol}. Falling back to daily data.")
            data = ticker.history(period=period, interval="1d")
            
        if data.empty:
            return None, None
            
        # Get current price (use last close if not available)
        try:
            live_price = ticker.info.get('regularMarketPrice', data['Close'].iloc[-1])
        except:
            live_price = data['Close'].iloc[-1] if not data.empty else None
            
        return data, live_price
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None, None

# Generate trading advice with OpenAI
def generate_advice(symbol, data, live_price, analysis_style, timeframe, risk_reward, custom_prompt=None):
    try:
        # Check if we have an OpenAI API key as an environment variable
        api_key = "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"  # Azure OpenAI API key
        base_url = "https://models.inference.ai.azure.com"  # Azure AI inference endpoint
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Select the appropriate analysis style prompt template
        if custom_prompt and custom_prompt.strip():
            prompt_template = custom_prompt
        else:
            prompt_template = ANALYSIS_STYLES.get(analysis_style, ANALYSIS_STYLES["Technical Analysis"])
        
        # Format the template with relevant data
        prompt = prompt_template.format(
            symbol=symbol,
            timeframe=timeframe,
            risk_reward=risk_reward,
            live_price=live_price,
            custom_prompt=custom_prompt if custom_prompt else ""
        )
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # The newest OpenAI model is "gpt-4o" which was released after May 2023
            messages=[
                {"role": "system", "content": "You are an expert financial analyst providing trading advice. Format your response nicely with bullet points, clear sections, and proper spacing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating advice: {e}")
        return f"""
## Trading Analysis Error
        
Unable to generate market analysis at this time. There might be an issue with the OpenAI API connection.

**Error details:** {str(e)}

Please try again later or select a different symbol.
"""

# Main app
def main():
    # Title
    st.markdown('<div class="title">Financial Advisor - Select or Search Markets</div>', unsafe_allow_html=True)
    
    # Display asset count stats if available
    if hasattr(st.session_state, 'asset_counts') and hasattr(st.session_state, 'total_assets'):
        with st.expander("📊 Asset Coverage Statistics"):
            total = st.session_state.total_assets
            counts = st.session_state.asset_counts
            
            st.markdown(f"**Total Assets Available:** {total}")
            st.markdown("---")
            cols = st.columns(3)
            
            # Display counts by category in columns
            for i, (category, count) in enumerate(counts.items()):
                if count > 0:  # Only show categories with assets
                    col_index = i % 3
                    cols[col_index].metric(category, count)
    
    # Create an empty categories dictionary to avoid reference errors
    categories = {}
    
    # Search box with enhanced UX
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("", placeholder="Search by symbol or company name (e.g., 'Reliance', 'Bitcoin', 'NIFTY', etc.)")
    
    # Load symbols
    symbols = load_symbols()
    symbol_options = [(k, v) for k, v in symbols.items()]
    symbol_options.sort(key=lambda x: x[1])  # Sort by name
    
    # Default to NIFTY 50
    default_symbol = "^NSEI"
    default_index = next((i for i, (k, v) in enumerate(symbol_options) if k == default_symbol), 0)
    
    # Filter symbols based on search with improved matching
    if search_query:
        # More granular search with partial matching and prioritization
        exact_matches = []
        symbol_starts_with = []
        name_starts_with = []
        symbol_contains = []
        name_contains = []
        
        search_lower = search_query.lower()
        
        for k, v in symbol_options:
            # Exact matches (highest priority)
            if k.lower() == search_lower or v.lower() == search_lower:
                exact_matches.append((k, v))
            # Symbol starts with search term (high priority)
            elif k.lower().startswith(search_lower):
                symbol_starts_with.append((k, v))
            # Name starts with search term (high priority)
            elif v.lower().startswith(search_lower):
                name_starts_with.append((k, v))
            # Symbol contains search term (medium priority)
            elif search_lower in k.lower():
                symbol_contains.append((k, v))
            # Name contains search term (medium priority)
            elif search_lower in v.lower():
                name_contains.append((k, v))
        
        # Combine results in priority order
        filtered_options = exact_matches + symbol_starts_with + name_starts_with + symbol_contains + name_contains
        
        # Limit to first 50 results for better performance
        filtered_options = filtered_options[:50]
        
        display_options = filtered_options
        
        if not filtered_options:
            st.warning(f"No symbols found matching '{search_query}'. Try using a different search term.")
            # Still provide all options
            display_options = symbol_options[:100]  # Limit to first 100
    else:
        display_options = symbol_options[:100]  # Limit to first 100 for performance
    
    # Create layout with labels and inputs
    col1, col2 = st.columns([1, 3])
    
    # Markets selection
    with col1:
        st.markdown('<div class="label">Markets</div>', unsafe_allow_html=True)
    
    with col2:
        if display_options:
            symbol_names = [f"{v} ({k})" for k, v in display_options]
            selected_symbol_name = st.selectbox(
                "",
                options=symbol_names,
                index=0,
                label_visibility="collapsed"
            )
            
            # Extract the actual symbol
            selected_symbol = display_options[symbol_names.index(selected_symbol_name)][0]
        else:
            selected_symbol = default_symbol
    
    # Strategy selection
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="label">Strategy</div>', unsafe_allow_html=True)
    with col2:
        selected_strategy = st.selectbox(
            "",
            options=list(STRATEGIES.keys()),
            index=0,
            label_visibility="collapsed"
        )
    
    # Timeframe selection (filtered based on selected strategy)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="label">Timeframe</div>', unsafe_allow_html=True)
    with col2:
        # Get available timeframes for the selected strategy
        strategy_config = STRATEGIES[selected_strategy]
        available_timeframes = strategy_config["timeframes"]
        
        selected_timeframe = st.selectbox(
            "",
            options=available_timeframes,
            index=0,
            label_visibility="collapsed"
        )
    
    # Risk-Reward selection
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="label">Risk-Reward</div>', unsafe_allow_html=True)
    with col2:
        selected_risk_reward = st.selectbox(
            "",
            options=RISK_REWARD_RATIOS,
            index=2,  # Default to 1:3
            label_visibility="collapsed"
        )
    
    # Analysis Style selection
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="label">Analysis Style</div>', unsafe_allow_html=True)
    with col2:
        selected_analysis_style = st.selectbox(
            "",
            options=list(ANALYSIS_STYLES.keys()),
            index=0,
            label_visibility="collapsed"
        )
    
    # Custom Prompt
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="label">Custom Prompt</div>', unsafe_allow_html=True)
    with col2:
        custom_prompt = st.text_area(
            "",
            placeholder="Enter your custom analysis prompt here (e.g., 'Analyze {symbol} using historical data...')",
            height=150,
            label_visibility="collapsed"
        )
    
    # Auto-select default button and Get Trading Advice button
    col1, col2 = st.columns([1, 1])
    with col1:
        auto_select_default = st.checkbox("Auto-select NIFTY 50 as default", value=True, 
                                        help="When checked, if no symbol is found in search, NIFTY 50 will be selected automatically")
    
    # Store the current advice in session state for sharing
    if "current_advice" not in st.session_state:
        st.session_state.current_advice = None
        st.session_state.current_symbol = None
        st.session_state.current_price = None
        st.session_state.current_analysis_style = None
    
    # Function to generate watermarked advice for sharing
    def get_watermarked_advice(advice, symbol, price, analysis_style):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        watermark = f"""
---
*Generated by Financial Trading Advisor on {timestamp}*
*Symbol: {symbol} | Price: {price:.2f} | Analysis: {analysis_style}*
*© 2025 Financial Trading Advisor - All rights reserved*
"""
        return advice + watermark
        
    # Function to extract trading recommendation summary
    def extract_recommendation_summary(advice):
        try:
            # Look for common patterns in the advice to extract key information
            summary = {}
            
            # Extract buy/sell recommendation
            if "BUY" in advice.upper():
                summary["signal"] = "BUY"
            elif "SELL" in advice.upper():
                summary["signal"] = "SELL"
            else:
                summary["signal"] = "NEUTRAL"
            
            # Extract confidence level - more flexible patterns
            confidence_patterns = [
                r'confidence[:\s]*(\d+)[%]?',
                r'confidence level[:\s]*(\d+)[%]?',
                r'confidence[:\s]*(\d+\.\d+)[%]?',
                r'confidence of[:\s]*(\d+)[%]?'
            ]
            
            confidence_found = False
            for pattern in confidence_patterns:
                confidence_match = re.search(pattern, advice.lower())
                if confidence_match:
                    confidence_val = confidence_match.group(1)
                    try:
                        confidence_num = float(confidence_val)
                        if confidence_num > 0:
                            summary["confidence"] = f"{int(confidence_num)}%"
                            confidence_found = True
                            break
                    except:
                        pass
            
            if not confidence_found:
                # Try to find any number followed by % that might be confidence
                percent_match = re.search(r'(\d+)%', advice)
                if percent_match:
                    summary["confidence"] = percent_match.group(0)
                else:
                    summary["confidence"] = "75%"  # Default value if nothing found
            
            # Extract entry price - more flexible patterns
            entry_patterns = [
                r'entry[:\s]*([\d\.]+)',
                r'entry price[:\s]*([\d\.]+)',
                r'entry at[:\s]*([\d\.]+)',
                r'enter at[:\s]*([\d\.]+)',
                r'enter[:\s]*([\d\.]+)'
            ]
            
            entry_found = False
            for pattern in entry_patterns:
                entry_match = re.search(pattern, advice.lower())
                if entry_match:
                    summary["entry"] = entry_match.group(1)
                    entry_found = True
                    break
            
            if not entry_found:
                summary["entry"] = "Current Price"
            
            # Extract take profit - more flexible patterns
            tp_patterns = [
                r'take[- ]*profit[:\s]*([\d\.]+)',
                r'take[- ]*profit[:\s]*target[:\s]*([\d\.]+)',
                r'tp[:\s]*([\d\.]+)',
                r'tp target[:\s]*([\d\.]+)',
                r'profit target[:\s]*([\d\.]+)'
            ]
            
            tp_found = False
            for pattern in tp_patterns:
                tp_match = re.search(pattern, advice.lower())
                if tp_match:
                    summary["tp"] = tp_match.group(1)
                    tp_found = True
                    break
            
            if not tp_found:
                # If no specific TP found, look for any number after Take Profit
                tp_section_match = re.search(r'take profit[^\.]*?([\d\.]+)', advice.lower())
                if tp_section_match:
                    summary["tp"] = tp_section_match.group(1)
                else:
                    summary["tp"] = "Target Pending"
            
            # Extract stop loss - more flexible patterns
            sl_patterns = [
                r'stop[- ]*loss[:\s]*([\d\.]+)',
                r'sl[:\s]*([\d\.]+)',
                r'stop[:\s]*([\d\.]+)',
                r'stop price[:\s]*([\d\.]+)'
            ]
            
            sl_found = False
            for pattern in sl_patterns:
                sl_match = re.search(pattern, advice.lower())
                if sl_match:
                    summary["sl"] = sl_match.group(1)
                    sl_found = True
                    break
            
            if not sl_found:
                # If no specific SL found, look for any number after Stop Loss
                sl_section_match = re.search(r'stop loss[^\.]*?([\d\.]+)', advice.lower())
                if sl_section_match:
                    summary["sl"] = sl_section_match.group(1)
                else:
                    summary["sl"] = "Stop Pending"
            
            return summary
        except Exception as e:
            print(f"Error extracting summary: {e}")
            # Fallback if parsing fails
            return {
                "signal": "CHECK ANALYSIS",
                "confidence": "75%",
                "entry": "Current Price",
                "tp": "Target Pending",
                "sl": "Stop Pending"
            }
    
    # Function to create recommendation image
    def create_recommendation_image(symbol, price, summary, analysis_style):
        try:
            # Create image with a more attractive design
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            
            # Create a plot with matplotlib - larger size for better quality
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Create a gradient background
            gradient = np.linspace(0, 1, 500).reshape(-1, 1)
            gradient = np.repeat(gradient, 500, 1)
            ax.imshow(gradient, aspect='auto', extent=[-1, 1, -1, 1], 
                     cmap=plt.get_cmap('gist_yarg'), alpha=0.8)
            
            # Add a dark overlay
            ax.add_patch(plt.Rectangle((-1, -1), 2, 2, facecolor='#1E1E1E', alpha=0.85))
            
            # Create a header bar
            ax.add_patch(plt.Rectangle((-1, 0.7), 2, 0.4, facecolor='#333333', alpha=0.7))
            
            # Set title with better styling
            title_bbox = dict(facecolor='#333333', alpha=0.7, pad=5)
            ax.text(0, 0.85, f"Trading Analysis for {symbol}", color='white', 
                   fontsize=22, ha='center', weight='bold', bbox=title_bbox)
            
            ax.text(0, 0.7, f"Current Price: {price:.2f}", color='white', fontsize=18, ha='center', weight='bold')
            
            # Signal text with appropriate color and better styling
            signal_color = '#4CAF50' if summary.get('signal') == 'BUY' else '#FF5252' if summary.get('signal') == 'SELL' else 'white'
            
            # Create a signal badge
            signal_text = summary.get('signal', 'CHECK')
            signal_badge = plt.Rectangle((-0.3, 0.45), 0.6, 0.15, facecolor=signal_color, alpha=0.2, 
                                     edgecolor=signal_color, linewidth=2)
            ax.add_patch(signal_badge)
            ax.text(0, 0.52, f"SIGNAL: {signal_text}", color=signal_color, fontsize=26, 
                   ha='center', weight='bold')
            
            # Add other information in an organized way
            ax.text(-0.45, 0.35, "CONFIDENCE", color='#888888', fontsize=14, ha='center')
            ax.text(-0.45, 0.28, f"{summary.get('confidence', '75%')}", color='white', fontsize=18, 
                   ha='center', weight='bold')
            
            ax.text(0, 0.35, "ENTRY", color='#888888', fontsize=14, ha='center')
            ax.text(0, 0.28, f"{summary.get('entry', 'Current Price')}", color='white', fontsize=18, 
                   ha='center', weight='bold')
            
            ax.text(0.45, 0.35, "TAKE PROFIT", color='#888888', fontsize=14, ha='center')
            ax.text(0.45, 0.28, f"{summary.get('tp', 'Target Pending')}", color='#4CAF50', fontsize=18, 
                   ha='center', weight='bold')
            
            ax.text(0, 0.15, "STOP LOSS", color='#888888', fontsize=14, ha='center')
            ax.text(0, 0.08, f"{summary.get('sl', 'Stop Pending')}", color='#FF5252', fontsize=18, 
                   ha='center', weight='bold')
            
            # Add a prominent Trading Hub logo/watermark in the center bottom
            # Create a text box with the company name
            logo_rect = plt.Rectangle((-0.6, -0.6), 1.2, 0.3, facecolor='#4CAF50', alpha=0.8, 
                                    edgecolor='white', linewidth=2)
            ax.add_patch(logo_rect)
            
            # Add the company name with a stroke effect
            from matplotlib.patheffects import withStroke
            ax.text(0, -0.45, "TRADING HUB", color='white', fontsize=28, 
                   ha='center', va='center', weight='bold', 
                   path_effects=[withStroke(linewidth=2, foreground='black')])
            
            # Add timestamp in smaller font below the logo
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            ax.text(0, -0.65, f"Generated: {timestamp} | Analysis: {analysis_style}", 
                   color='white', fontsize=10, ha='center', alpha=0.7)
            
            # Hide axes and set tight layout
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # Convert plot to an image with high quality
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=200, facecolor='#1E1E1E', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
        except Exception as e:
            st.error(f"Error creating image: {e}")
            print(f"Error creating image: {e}")
            # Try a simpler fallback image if the first attempt fails
            try:
                # Create a simpler plot
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#1E1E1E')
                ax.set_facecolor('#1E1E1E')
                
                # Define signal color for the fallback
                signal_text = summary.get('signal', 'CHECK')
                fallback_signal_color = '#4CAF50' if signal_text == 'BUY' else '#FF5252' if signal_text == 'SELL' else 'white'
                
                # Set title and basic content
                ax.text(0.5, 0.9, "TRADING HUB", color='#4CAF50', fontsize=30, ha='center', 
                       weight='bold', alpha=1.0)
                ax.text(0.5, 0.75, f"Analysis for {symbol}", color='white', fontsize=18, ha='center')
                ax.text(0.5, 0.65, f"Signal: {signal_text}", color=fallback_signal_color, fontsize=22, 
                       ha='center', weight='bold')
                ax.text(0.5, 0.55, f"Price: {price:.2f}", color='white', fontsize=16, ha='center')
                
                # Add confidence, TP, SL directly to the fallback image
                ax.text(0.5, 0.45, f"Confidence: {summary.get('confidence', '75%')}", color='white', fontsize=16, ha='center')
                ax.text(0.5, 0.35, f"Take Profit: {summary.get('tp', 'Target Pending')}", color='#4CAF50', fontsize=16, ha='center')
                ax.text(0.5, 0.25, f"Stop Loss: {summary.get('sl', 'Stop Pending')}", color='#FF5252', fontsize=16, ha='center')
                
                # Add analysis style
                ax.text(0.5, 0.15, f"Analysis: {analysis_style}", color='gray', fontsize=12, ha='center', alpha=0.7)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                ax.text(0.5, 0.05, f"Generated: {timestamp}", color='gray', fontsize=10, ha='center', alpha=0.7)
                
                # Hide axes
                ax.axis('off')
                
                # Convert plot to an image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, facecolor='#1E1E1E')
                buf.seek(0)
                
                # Convert to base64
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                
                return img_str
            except Exception as e2:
                print(f"Even fallback image failed: {e2}")
                return None
    
    # Get Trading Advice button
    if st.button("Get Trading Advice", use_container_width=True):
        # If no symbol is selected and auto-select is enabled, use default
        if not selected_symbol and auto_select_default:
            selected_symbol = "^NSEI"  # Default to NIFTY 50
            st.info("Auto-selected NIFTY 50 as default symbol")
        
        with st.spinner("Analyzing market data and calculating technical indicators..."):
            # Fetch data
            timeframe_code = TIMEFRAMES[selected_timeframe]
            
            # Get appropriate days limit from strategy
            strategy_days = strategy_config["days"]
            
            # Fetch market data with the appropriate days limit for the strategy
            data, live_price = fetch_market_data(selected_symbol, timeframe=timeframe_code, days=strategy_days)
            
            if data is not None and not data.empty and live_price is not None:
                # Calculate all technical indicators using our advanced indicators module
                from utils.indicators import calculate_indicators
                
                # Apply indicators to the data
                with st.spinner("Calculating 30+ advanced technical indicators..."):
                    try:
                        data_with_indicators = calculate_indicators(data)
                        
                        # Get indicator summary 
                        from utils.indicators import get_indicator_summary, format_indicator_data
                        indicator_summary = get_indicator_summary(data_with_indicators)
                        formatted_indicators = format_indicator_data(data_with_indicators)
                        
                        # Add indicator information to prompt
                        indicator_info = ", ".join([f"{k}: {v}" for k, v in indicator_summary.items() if not isinstance(v, dict) and not k.endswith('_Value')])
                    except Exception as e:
                        st.warning(f"Some indicators couldn't be calculated due to data limitations, but the analysis will still continue.")
                        indicator_info = "Limited indicator data available"
                
                # Generate advice
                advice = generate_advice(
                    selected_symbol,
                    data,
                    live_price,
                    selected_analysis_style,
                    selected_timeframe,
                    selected_risk_reward,
                    custom_prompt
                )
                
                # Store current advice in session state
                st.session_state.current_advice = advice
                st.session_state.current_symbol = symbols.get(selected_symbol, selected_symbol)
                st.session_state.current_price = live_price
                st.session_state.current_analysis_style = selected_analysis_style
                
                # Display results header
                st.subheader(f"Trading Analysis for {symbols.get(selected_symbol, selected_symbol)}")
                st.markdown(f"**Current Price:** {live_price:.2f}")
                
                # Extract recommendation summary for social sharing
                summary = extract_recommendation_summary(advice)
                
                # Create recommendation image
                recommendation_image = create_recommendation_image(
                    symbols.get(selected_symbol, selected_symbol),
                    live_price,
                    summary,
                    selected_analysis_style
                )
                
                # Social sharing section
                st.markdown("### Share this analysis")
                share_col1, share_col2 = st.columns([3, 1])
                
                with share_col1:
                    # Store recommendation image in session state for sharing
                    if recommendation_image and "recommendation_image" not in st.session_state:
                        st.session_state.recommendation_image = recommendation_image
                        
                    # Display the recommendation image
                    if recommendation_image:
                        st.image(f"data:image/png;base64,{recommendation_image}", caption="Trading Recommendation Summary")
                
                with share_col2:
                    # Create sharing button
                    symbol_name = symbols.get(selected_symbol, selected_symbol)
                    share_text = f"Trading Analysis for {symbol_name}: {summary.get('signal', 'N/A')} with {summary.get('confidence', 'N/A')} confidence. #FinancialTrading #MarketAnalysis"
                    
                    if st.button("📤 Share Recommendation", key="share_button", use_container_width=True):
                        # Create a data URL for the image that can be shared
                        if recommendation_image:
                            # Create watermarked version of summary text
                            watermarked_summary = f"""
                            TRADING HUB
                            Trading Analysis for {symbol_name}
                            Signal: {summary.get('signal', 'N/A')} | Confidence: {summary.get('confidence', 'N/A')}
                            Entry: {summary.get('entry', 'N/A')} | TP: {summary.get('tp', 'N/A')} | SL: {summary.get('sl', 'N/A')}
                            
                            Generated by Trading Hub on {datetime.now().strftime("%Y-%m-%d")}
                            """
                            
                            # Use JavaScript to copy the summary and notify the user that image is ready to share
                            js = f"""
                            <script>
                            const el = document.createElement('textarea');
                            el.value = {repr(watermarked_summary)};
                            document.body.appendChild(el);
                            el.select();
                            document.execCommand('copy');
                            document.body.removeChild(el);
                            
                            // Create a temporary link to download the image
                            const link = document.createElement('a');
                            link.href = 'data:image/png;base64,{recommendation_image}';
                            link.download = 'trading_recommendation.png';
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            alert('Trading recommendation copied to clipboard and image downloaded! You can now share the image on any platform.');
                            </script>
                            """
                            st.components.v1.html(js, height=0)
                        else:
                            st.error("Could not generate recommendation image for sharing.")
                
                # Display the advice
                st.markdown("---")
                st.markdown(advice)
                
            else:
                st.error(f"Failed to retrieve data for {selected_symbol}. Please try another symbol or timeframe.")

if __name__ == "__main__":
    main()