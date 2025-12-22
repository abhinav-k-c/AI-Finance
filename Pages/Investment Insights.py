import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date, timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our new ML utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from stock_ml_utils import (
    StockDataManager, 
    train_model_for_stock, 
    predict_stock_price,
    STOCK_SYMBOLS
)

# Initialize session state
def initialize_investment_session():
    """Initialize investment session state"""
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    if 'investment_profile' not in st.session_state:
        st.session_state.investment_profile = {
            'risk_tolerance': 'Moderate',
            'investment_horizon': '5-10 years',
            'investment_amount': 10000
        }

def get_combined_financial_data():
    """Get combined financial data for investment recommendations"""
    combined_df = pd.DataFrame()
    
    # Get manual entries
    if 'manual_entries' in st.session_state and st.session_state.manual_entries:
        manual_df = pd.DataFrame(st.session_state.manual_entries)
        combined_df = pd.concat([combined_df, manual_df], ignore_index=True)
    
    # Get uploaded data
    if 'uploaded_data' in st.session_state:
        uploaded_df = st.session_state.uploaded_data.copy()
        uploaded_df.columns = uploaded_df.columns.str.lower()
        combined_df = pd.concat([combined_df, uploaded_df], ignore_index=True)
    
    if not combined_df.empty:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df['amount'] = pd.to_numeric(combined_df['amount'])
        combined_df['type'] = combined_df['type'].str.lower()
        combined_df = combined_df.sort_values('date')
    
    return combined_df

def get_market_data(symbol, period="1y"):
    """Fetch market data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def get_indian_stocks_data():
    """Get major Indian stock indices data"""
    indian_symbols = {
        '^NSEI': 'Nifty 50',
        '^BSESN': 'Sensex',
        '^NSEBANK': 'Bank Nifty',
        'RELIANCE.NS': 'Reliance',
        'TCS.NS': 'TCS',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'ICICIBANK.NS': 'ICICI Bank',
        'KOTAKBANK.NS': 'Kotak Bank',
        'LT.NS': 'L&T',
        'SBIN.NS': 'SBI',
        'BHARTIARTL.NS': 'Bharti Airtel'
    }
    
    market_data = {}
    for symbol, name in indian_symbols.items():
        data = get_market_data(symbol, period="5d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            market_data[symbol] = {
                'name': name,
                'price': current_price,
                'change': change,
                'change_pct': change_pct,
                'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
            }
    
    return market_data

def get_commodity_data():
    """Get commodity prices (Gold, Silver, Crude Oil)"""
    commodities = {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'CL=F': 'Crude Oil',
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum'
    }
    
    commodity_data = {}
    for symbol, name in commodities.items():
        data = get_market_data(symbol, period="5d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            commodity_data[symbol] = {
                'name': name,
                'price': current_price,
                'change': change,
                'change_pct': change_pct
            }
    
    return commodity_data

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    if data is None or data.empty:
        return {}
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    return data

def predict_price_trend(data, days_ahead=30):
    """Simple price prediction using linear regression"""
    if data is None or data.empty or len(data) < 30:
        return None, None, "Insufficient data"
    
    try:
        # Prepare features
        data_reset = data.reset_index()
        data_reset['Days'] = range(len(data_reset))
        
        # Use last 100 days for prediction
        recent_data = data_reset.tail(100).copy()
        
        if len(recent_data) < 10:
            return None, None, "Insufficient recent data"
        
        # Features: Days, Volume, High, Low
        features = ['Days']
        if 'Volume' in recent_data.columns:
            features.append('Volume')
        
        X = recent_data[features].fillna(method='ffill').fillna(0)
        y = recent_data['Close'].fillna(method='ffill')
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future prices
        future_days = []
        last_day = recent_data['Days'].iloc[-1]
        avg_volume = recent_data['Volume'].mean() if 'Volume' in recent_data.columns else 0
        
        predictions = []
        for i in range(1, days_ahead + 1):
            future_day = last_day + i
            future_features = [[future_day]]
            if 'Volume' in features:
                future_features[0].append(avg_volume)
            
            pred = model.predict(future_features)
            predictions.append(float(pred[0]))  # Extract scalar value from numpy array
            future_days.append(future_day)
        
        # Determine trend
        current_price = float(recent_data['Close'].iloc[-1])
        future_price = predictions[-1]
        trend = "Bullish" if future_price > current_price else "Bearish"
        confidence = min(abs((future_price - current_price) / current_price) * 100, 85)
        
        return predictions, future_days, f"{trend} (Confidence: {confidence:.1f}%)"
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

def create_price_chart(data, symbol, predictions=None, future_days=None):
    """Create interactive price chart with technical indicators"""
    if data is None or data.empty:
        return None
    
    # Calculate technical indicators
    data_with_indicators = calculate_technical_indicators(data.copy())
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Chart', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data_with_indicators.index,
            open=data_with_indicators['Open'],
            high=data_with_indicators['High'],
            low=data_with_indicators['Low'],
            close=data_with_indicators['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.5
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.5,
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ),
        row=1, col=1
    )
    
    # Predictions
    if predictions and future_days:
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, len(predictions) + 1)]
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                name='Prediction',
                line=dict(color='red', width=2, dash='dot')
            ),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD'],
            name='MACD',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD_Signal'],
            name='Signal',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis3_title="Date",
        yaxis_title="Price",
        yaxis2_title="RSI",
        yaxis3_title="MACD"
    )
    
    return fig

def market_overview_dashboard():
    """Market overview dashboard"""
    st.subheader("📊 Market Overview")
    
    # Fetch market data
    with st.spinner("Fetching market data..."):
        market_data = get_indian_stocks_data()
        commodity_data = get_commodity_data()
    
    if not market_data and not commodity_data:
        st.error("Unable to fetch market data. Please check your internet connection.")
        return
    
    # Market indices
    if market_data:
        st.write("**📈 Major Indian Indices & Stocks**")
        
        indices = ['^NSEI', '^BSESN', '^NSEBANK']
        index_data = {k: v for k, v in market_data.items() if k in indices}
        
        if index_data:
            cols = st.columns(len(index_data))
            for i, (symbol, data) in enumerate(index_data.items()):
                with cols[i]:
                    delta_color = "normal" if data['change'] >= 0 else "inverse"
                    st.metric(
                        data['name'],
                        f"₹{data['price']:,.2f}",
                        f"{data['change']:+,.2f} ({data['change_pct']:+.2f}%)",
                        delta_color=delta_color
                    )
        
        # Top stocks
        st.write("**🏢 Top Stocks**")
        stock_data = {k: v for k, v in market_data.items() if k not in indices}
        
        if stock_data:
            # Create DataFrame with proper structure
            stocks_list = []
            for symbol, data in stock_data.items():
                stocks_list.append({
                    'Symbol': symbol,
                    'Name': data['name'],
                    'Price': round(float(data['price']), 2),
                    'Change': round(float(data['change']), 2),
                    'Change %': round(float(data['change_pct']), 2),
                    'Volume': int(data['volume']) if data['volume'] else 0
                })
            
            stocks_df = pd.DataFrame(stocks_list)
            
            # Display the DataFrame
            display_df = stocks_df[['Name', 'Price', 'Change', 'Change %']].head(10)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Commodities and Crypto
    if commodity_data:
        st.markdown("---")
        st.write("**💰 Commodities & Cryptocurrencies**")
        
        cols = st.columns(min(len(commodity_data), 5))
        for i, (symbol, data) in enumerate(commodity_data.items()):
            if i < 5:  # Limit to 5 columns
                with cols[i]:
                    delta_color = "normal" if data['change'] >= 0 else "inverse"
                    price_format = f"${data['price']:,.2f}" if 'USD' in symbol or symbol in ['GC=F', 'SI=F', 'CL=F'] else f"₹{data['price']:,.2f}"
                    st.metric(
                        data['name'],
                        price_format,
                        f"{data['change']:+,.2f} ({data['change_pct']:+.2f}%)",
                        delta_color=delta_color
                    )

def stock_analysis_tool():
    """Stock analysis and prediction tool - ML POWERED"""
    st.subheader("🔍 Stock Analysis & ML Predictions")
    
    # Info banner about ML approach
    st.info("🤖 **Powered by XGBoost ML Model** | Data cached for 2 weeks | 20+ technical indicators")
    
    # Stock search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Use stocks from STOCK_SYMBOLS
        selected_stock = st.selectbox(
            "Select Stock for Analysis",
            options=list(STOCK_SYMBOLS.keys()),
            format_func=lambda x: f"{STOCK_SYMBOLS[x]} ({x})"
        )
    
    with col2:
        time_period = st.selectbox("Time Period (Display)", ["1y", "6mo", "3mo", "1mo"])
    
    with col3:
        prediction_days = st.number_input("Prediction Days", min_value=7, max_value=90, value=30, step=1)
    
    # Training controls
    col1, col2 = st.columns(2)
    with col1:
        train_button = st.button("🎓 Train/Retrain Model", type="secondary")
    with col2:
        analyze_button = st.button("📊 Analyze & Predict", type="primary")
    
    # Train model if requested
    if train_button:
        with st.spinner(f"🤖 Training ML model for {STOCK_SYMBOLS[selected_stock]}... This may take 30-60 seconds..."):
            success = train_model_for_stock(selected_stock, force_refresh=True)
            if success:
                st.success(f"✅ Model trained successfully for {STOCK_SYMBOLS[selected_stock]}!")
                st.balloons()
            else:
                st.error("❌ Model training failed. Please check logs.")
    
    # Analyze and predict
    if analyze_button:
        with st.spinner(f"🔮 Analyzing {STOCK_SYMBOLS[selected_stock]} with ML model..."):
            # Get stock data for display
            data_manager = StockDataManager()
            stock_data = data_manager.get_stock_data(selected_stock)
            
            if stock_data is None or stock_data.empty:
                st.error("❌ Unable to fetch stock data. Please try again.")
                return
            
            # Get ML prediction
            prediction_result, error = predict_stock_price(selected_stock, days_ahead=prediction_days)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 **Tip:** Click '🎓 Train/Retrain Model' button above to train the model first!")
                return
            
            # Display current metrics
            current_price = prediction_result['current_price']
            predicted_price = prediction_result['predicted_price']
            change = prediction_result['change']
            change_pct = prediction_result['change_pct']
            trend = prediction_result['trend']
            confidence = prediction_result['confidence']
            
            # Get historical change
            prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
            daily_change = current_price - prev_price
            daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{current_price:,.2f}")
            
            with col2:
                delta_color = "normal" if daily_change >= 0 else "inverse"
                st.metric("Daily Change", f"₹{daily_change:+,.2f}", f"{daily_change_pct:+.2f}%", delta_color=delta_color)
            
            with col3:
                volume = stock_data['Volume'].iloc[-1] if 'Volume' in stock_data.columns else 0
                st.metric("Volume", f"{volume:,.0f}")
            
            with col4:
                high_52w = stock_data['High'].max()
                st.metric("52W High", f"₹{high_52w:,.2f}")
            
            # ML Prediction Results
            st.markdown("---")
            st.subheader(f"🔮 ML Prediction ({prediction_days} days ahead)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Predicted Price",
                    f"₹{predicted_price:,.2f}",
                    f"{change:+,.2f} ({change_pct:+.2f}%)",
                    delta_color="normal" if change > 0 else "inverse"
                )
            
            with col2:
                trend_color = "🟢" if "Bullish" in trend else "🔴"
                st.metric("Trend", f"{trend_color} {trend}")
            
            with col3:
                confidence_emoji = "🎯" if confidence > 70 else "⚠️"
                st.metric("Confidence", f"{confidence_emoji} {confidence:.1f}%")
            
            # Price range (confidence interval)
            price_range_low = predicted_price * 0.95
            price_range_high = predicted_price * 1.05
            
            st.info(f"📊 **Predicted Price Range:** ₹{price_range_low:,.2f} - ₹{price_range_high:,.2f} (±5%)")
            
            # Display chart with historical data (for visualization)
            display_data = get_market_data(selected_stock, time_period)
            if display_data is not None:
                st.markdown("---")
                st.subheader("📈 Price Chart & Technical Indicators")
                
                # Use old chart function for visualization
                data_with_indicators = calculate_technical_indicators(display_data.copy())
                if not data_with_indicators.empty:
                    # Simple price chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=display_data.index,
                        open=display_data['Open'],
                        high=display_data['High'],
                        low=display_data['Low'],
                        close=display_data['Close'],
                        name='Price'
                    ))
                    
                    # Add prediction point
                    future_date = display_data.index[-1] + timedelta(days=prediction_days)
                    fig.add_trace(go.Scatter(
                        x=[display_data.index[-1], future_date],
                        y=[current_price, predicted_price],
                        mode='lines+markers',
                        name=f'{prediction_days}d Prediction',
                        line=dict(color='red', width=3, dash='dot'),
                        marker=dict(size=10, color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"{STOCK_SYMBOLS[selected_stock]} - Price & ML Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical signals
                    st.subheader("📊 Technical Signals")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        rsi = data_with_indicators['RSI'].iloc[-1]
                        if rsi > 70:
                            st.warning(f"RSI: {rsi:.1f} - Overbought")
                        elif rsi < 30:
                            st.success(f"RSI: {rsi:.1f} - Oversold")
                        else:
                            st.info(f"RSI: {rsi:.1f} - Neutral")
                    
                    with col2:
                        sma_20 = data_with_indicators['SMA_20'].iloc[-1]
                        sma_50 = data_with_indicators['SMA_50'].iloc[-1]
                        if current_price > sma_20 > sma_50:
                            st.success("Moving Averages: Bullish")
                        elif current_price < sma_20 < sma_50:
                            st.error("Moving Averages: Bearish")
                        else:
                            st.info("Moving Averages: Mixed")
                    
                    with col3:
                        macd = data_with_indicators['MACD'].iloc[-1]
                        macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
                        if macd > macd_signal:
                            st.success("MACD: Bullish Signal")
                        else:
                            st.error("MACD: Bearish Signal")
            
            # Add to watchlist
            st.markdown("---")
            if st.button(f"⭐ Add {STOCK_SYMBOLS[selected_stock]} to Watchlist"):
                watchlist_item = {
                    'symbol': selected_stock,
                    'name': STOCK_SYMBOLS[selected_stock],
                    'added_date': datetime.now().strftime('%Y-%m-%d'),
                    'added_price': current_price
                }
                
                if selected_stock not in [item['symbol'] for item in st.session_state.watchlist]:
                    st.session_state.watchlist.append(watchlist_item)
                    st.success("✅ Added to watchlist!")
                else:
                    st.info("ℹ️ Already in watchlist!")

def investment_recommendations():
    """AI-powered investment recommendations"""
    st.subheader("💡 Personalized Investment Recommendations")
    
    # Get user's financial data
    df = get_combined_financial_data()
    
    # Investment profile setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**👤 Investment Profile**")
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ['Conservative', 'Moderate', 'Aggressive'],
            index=1
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ['< 1 year', '1-3 years', '3-5 years', '5-10 years', '> 10 years'],
            index=2
        )
        
        monthly_investment = st.number_input(
            "Monthly Investment Amount (₹)",
            min_value=500,
            max_value=100000,
            value=5000,
            step=500
        )
    
    with col2:
        st.write("**💰 Financial Summary**")
        if not df.empty:
            income = df[df['type'] == 'income']['amount'].sum()
            expenses = df[df['type'] == 'expense']['amount'].sum()
            savings = income - expenses
            
            st.write(f"• **Total Income:** ₹{income:,.2f}")
            st.write(f"• **Total Expenses:** ₹{expenses:,.2f}")
            st.write(f"• **Net Savings:** ₹{savings:,.2f}")
            
            if income > 0:
                savings_rate = (savings / income) * 100
                st.write(f"• **Savings Rate:** {savings_rate:.1f}%")
        else:
            st.info("💡 Add financial data for personalized recommendations")
    
    # Generate recommendations
    if st.button("🎯 Generate Investment Recommendations", type="primary"):
        st.markdown("---")
        st.subheader("📋 Your Personalized Investment Plan")
        
        # Asset allocation based on risk profile
        if risk_tolerance == 'Conservative':
            allocation = {'Debt': 60, 'Equity': 30, 'Gold': 10}
            expected_return = '6-8%'
        elif risk_tolerance == 'Moderate':
            allocation = {'Equity': 50, 'Debt': 40, 'Gold': 10}
            expected_return = '8-12%'
        else:  # Aggressive
            allocation = {'Equity': 70, 'Debt': 20, 'Gold': 10}
            expected_return = '10-15%'
        
        # Display allocation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🥧 Recommended Asset Allocation:**")
            for asset, percentage in allocation.items():
                amount = (monthly_investment * percentage) / 100
                st.write(f"• **{asset}:** {percentage}% (₹{amount:,.0f}/month)")
            
            st.success(f"**Expected Annual Return:** {expected_return}")
        
        with col2:
            # Pie chart for allocation
            fig = px.pie(
                values=list(allocation.values()),
                names=list(allocation.keys()),
                title="Asset Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Specific investment recommendations
        st.write("**🎯 Specific Investment Options:**")
        
        recommendations = {
            'Equity': [
                'Nifty 50 Index Fund - Low cost, diversified',
                'Large Cap Mutual Fund - Stable growth',
                'ELSS Fund - Tax saving + growth'
            ],
            'Debt': [
                'Liquid Funds - Emergency fund parking',
                'Short Duration Funds - 1-3 year goals',
                'PPF - Long term tax saving'
            ],
            'Gold': [
                'Gold ETF - Easy liquidity',
                'Gold Mutual Fund - Systematic investment',
                'Digital Gold - Small amounts'
            ]
        }
        
        for category, options in recommendations.items():
            st.write(f"**{category} Options:**")
            for option in options:
                st.write(f"  • {option}")
        
        # SIP calculation
        st.markdown("---")
        st.write("**📈 SIP Projection:**")
        
        # Simple SIP calculation assuming average returns
        if risk_tolerance == 'Conservative':
            annual_return = 0.07
        elif risk_tolerance == 'Moderate':
            annual_return = 0.10
        else:
            annual_return = 0.12
        
        years = int(investment_horizon.split('-')[0]) if '-' in investment_horizon else 10
        months = years * 12
        monthly_return = annual_return / 12
        
        # Future value calculation
        fv = monthly_investment * (((1 + monthly_return) ** months - 1) / monthly_return)
        invested_amount = monthly_investment * months
        returns = fv - invested_amount
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Investment", f"₹{invested_amount:,.0f}")
        
        with col2:
            st.metric("Expected Returns", f"₹{returns:,.0f}")
        
        with col3:
            st.metric("Final Amount", f"₹{fv:,.0f}")

def portfolio_tracker():
    """Portfolio tracking and management"""
    st.subheader("💼 Portfolio Tracker")
    
    # Add investment to portfolio
    with st.expander("➕ Add Investment to Portfolio"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment_name = st.text_input("Investment Name", placeholder="e.g., Nifty 50 ETF")
            investment_type = st.selectbox("Type", ["Mutual Fund", "Stock", "ETF", "FD", "PPF", "Other"])
        
        with col2:
            units = st.number_input("Units/Quantity", min_value=0.01, step=0.01, value=1.0)
            buy_price = st.number_input("Buy Price (₹)", min_value=0.01, step=0.01, value=100.0)
        
        with col3:
            buy_date = st.date_input("Purchase Date", value=date.today())
            current_price = st.number_input("Current Price (₹)", min_value=0.01, step=0.01, value=buy_price)
        
        if st.button("📊 Add to Portfolio"):
            if investment_name:
                portfolio_item = {
                    'name': investment_name,
                    'type': investment_type,
                    'units': units,
                    'buy_price': buy_price,
                    'buy_date': buy_date.isoformat(),
                    'current_price': current_price,
                    'invested_amount': units * buy_price,
                    'current_value': units * current_price,
                    'profit_loss': (units * current_price) - (units * buy_price),
                    'profit_loss_pct': ((current_price - buy_price) / buy_price) * 100
                }
                
                st.session_state.portfolio.append(portfolio_item)
                st.success(f"Added {investment_name} to portfolio!")
                st.rerun()
    
    # Display portfolio
    if st.session_state.portfolio:
        st.write("**📊 Your Portfolio:**")
        
        # Portfolio summary
        total_invested = sum([item['invested_amount'] for item in st.session_state.portfolio])
        total_current = sum([item['current_value'] for item in st.session_state.portfolio])
        total_pl = total_current - total_invested
        total_pl_pct = (total_pl / total_invested) * 100 if total_invested > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        
        with col2:
            st.metric("Current Value", f"₹{total_current:,.2f}")
        
        with col3:
            delta_color = "normal" if total_pl >= 0 else "inverse"
            st.metric("Profit/Loss", f"₹{total_pl:,.2f}", f"{total_pl_pct:+.2f}%", delta_color=delta_color)
        
        with col4:
            avg_return = total_pl_pct
            st.metric("Avg Return", f"{avg_return:.2f}%")
        
        # Portfolio table
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        portfolio_display = portfolio_df[['name', 'type', 'units', 'buy_price', 'current_price', 'invested_amount', 'current_value', 'profit_loss', 'profit_loss_pct']]
        portfolio_display.columns = ['Investment', 'Type', 'Units', 'Buy Price', 'Current Price', 'Invested', 'Current Value', 'P&L', 'P&L %']
        
        st.dataframe(portfolio_display, use_container_width=True, hide_index=True)
        
        # Portfolio allocation chart
        allocation_data = portfolio_df.groupby('type')['current_value'].sum().reset_index()
        if not allocation_data.empty:
            fig = px.pie(
                allocation_data,
                values='current_value',
                names='type',
                title="Portfolio Allocation by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Clear portfolio option
        if st.button("🗑️ Clear Portfolio", type="secondary"):
            st.session_state.portfolio = []
            st.success("Portfolio cleared!")
            st.rerun()
    else:
        st.info("📝 Your portfolio is empty. Add some investments to get started!")

def watchlist_management():
    """Manage investment watchlist"""
    st.subheader("⭐ Investment Watchlist")
    
    if st.session_state.watchlist:
        st.write(f"**📋 You're watching {len(st.session_state.watchlist)} investments:**")
        
        for i, item in enumerate(st.session_state.watchlist):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{item['name']}** ({item['symbol']})")
                st.write(f"Added: {item['added_date']} at ₹{item['added_price']:.2f}")
            
            with col2:
                # Get current price
                current_data = get_market_data(item['symbol'], period="1d")
                if current_data is not None and not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    change = current_price - item['added_price']
                    change_pct = (change / item['added_price']) * 100
                    
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric("Current", f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)", delta_color=delta_color)
                else:
                    st.write("Price unavailable")
            
            with col3:
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.watchlist.pop(i)
                    st.success("Removed from watchlist!")
                    st.rerun()
        
        # Clear all watchlist
        if st.button("🗑️ Clear All Watchlist", type="secondary"):
            st.session_state.watchlist = []
            st.success("Watchlist cleared!")
            st.rerun()
    else:
        st.info("⭐ Your watchlist is empty. Use the Stock Analysis tool to add stocks to your watchlist!")

def main():
    """Main function for Investment Insights page"""
    
    st.title("📈 Investment Insights & Analysis")
    st.markdown("---")
    
    # Initialize session state
    initialize_investment_session()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", 
        "🔍 Stock Analysis", 
        "💡 Recommendations", 
        "💼 Portfolio", 
        "⭐ Watchlist"
    ])
    
    with tab1:
        market_overview_dashboard()
    
    with tab2:
        stock_analysis_tool()
    
    with tab3:
        investment_recommendations()
    
    with tab4:
        portfolio_tracker()
    
    with tab5:
        watchlist_management()

if __name__ == "__main__":
    main()
