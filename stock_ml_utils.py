"""
Stock ML Utilities - Data Management, Feature Engineering, and Model Training
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
from ta import add_all_ta_features
from ta.utils import dropna
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Directories
DATA_DIR = "data/stock_history"
MODEL_DIR = "models"
METADATA_DIR = "model_metadata"

# Cache expiry: 2-3 weeks (let's use 2 weeks = 14 days)
CACHE_EXPIRY_DAYS = 14

# Stock symbols to track
STOCK_SYMBOLS = {
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


class StockDataManager:
    """Manages stock data fetching, caching, and expiry checks"""
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(METADATA_DIR, exist_ok=True)
    
    def get_data_filepath(self, symbol):
        """Get filepath for cached data"""
        safe_symbol = symbol.replace('/', '_')
        return os.path.join(DATA_DIR, f"{safe_symbol}_data.csv")
    
    def get_metadata_filepath(self, symbol):
        """Get filepath for metadata"""
        safe_symbol = symbol.replace('/', '_')
        return os.path.join(METADATA_DIR, f"{safe_symbol}_metadata.json")
    
    def is_cache_expired(self, symbol):
        """Check if cached data has expired (older than 14 days)"""
        metadata_file = self.get_metadata_filepath(symbol)
        
        if not os.path.exists(metadata_file):
            return True
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            last_updated = datetime.fromisoformat(metadata.get('last_updated', '2000-01-01'))
            days_old = (datetime.now() - last_updated).days
            
            return days_old >= CACHE_EXPIRY_DAYS
        except:
            return True
    
    def fetch_and_cache_data(self, symbol, period='3y', force_refresh=False):
        """
        Fetch stock data and cache it locally
        Returns: DataFrame with historical data
        """
        data_file = self.get_data_filepath(symbol)
        metadata_file = self.get_metadata_filepath(symbol)
        
        # Check if we need to refresh
        if not force_refresh and not self.is_cache_expired(symbol):
            # Use cached data
            if os.path.exists(data_file):
                print(f"✅ Using cached data for {symbol}")
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                return df
        
        # Fetch fresh data from API
        print(f"🔄 Fetching fresh data for {symbol} (period: {period})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                return None
            
            # Save to CSV
            df.to_csv(data_file)
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'last_updated': datetime.now().isoformat(),
                'period': period,
                'rows': len(df),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Cached {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_data(self, symbol):
        """Get stock data (from cache or fetch fresh if expired)"""
        return self.fetch_and_cache_data(symbol, period='3y')


class FeatureEngineer:
    """Calculate technical indicators and prepare features for ML"""
    
    @staticmethod
    def add_technical_indicators(df):
        """Add 20+ technical indicators using ta library"""
        if df is None or df.empty:
            return None
        
        try:
            # Make a clean copy
            df = df.copy()
            
            # Reset index to avoid iloc issues
            df = df.reset_index(drop=False)
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # Clean data - remove rows with NaN
            df = df.dropna()
            
            if len(df) < 100:
                print("❌ Insufficient data after cleaning")
                return None
            
            # Add all technical indicators
            df = add_all_ta_features(
                df, 
                open="Open", 
                high="High", 
                low="Low", 
                close="Close", 
                volume="Volume",
                fillna=True
            )
            
            # Add custom features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Price momentum
            df['momentum_5'] = df['Close'] - df['Close'].shift(5)
            df['momentum_10'] = df['Close'] - df['Close'].shift(10)
            df['momentum_20'] = df['Close'] - df['Close'].shift(20)
            
            # Volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # Drop NaN values created by indicators
            df = df.dropna()
            
            if len(df) < 50:
                print("❌ Insufficient data after feature engineering")
                return None
            
            print(f"✅ Successfully added {len(df.columns)} features to {len(df)} rows")
            
            return df
            
        except Exception as e:
            print(f"❌ Error adding technical indicators: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def prepare_features(df, target_days=30):
        """Prepare features and target for model training"""
        if df is None or df.empty:
            return None, None, None, None, None
        
        # Create target: Future closing price (shifted by target_days)
        df['target'] = df['Close'].shift(-target_days)
        
        # Drop rows with NaN target
        df = df.dropna()
        
        if len(df) < 100:
            print("❌ Insufficient data for training")
            return None, None, None, None, None
        
        # Select features (exclude target and non-numeric columns)
        exclude_cols = ['target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get numeric columns only
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].values
        y = df['target'].values
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, numeric_cols


class StockMLModel:
    """XGBoost-based stock prediction model"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = None
        
    def get_model_filepath(self):
        """Get filepath for saved model"""
        safe_symbol = self.symbol.replace('/', '_')
        return os.path.join(MODEL_DIR, f"{safe_symbol}_model.pkl")
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train XGBoost model"""
        print(f"🤖 Training model for {self.symbol}...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            print(f"✅ Model trained - MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}")
            
            return {'mae': float(mae), 'rmse': float(rmse)}
        
        return None
    
    def save(self, metrics=None):
        """Save model and metadata"""
        model_file = self.get_model_filepath()
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        joblib.dump(model_data, model_file)
        print(f"💾 Model saved: {model_file}")
    
    def load(self):
        """Load trained model"""
        model_file = self.get_model_filepath()
        
        if not os.path.exists(model_file):
            return False
        
        try:
            model_data = joblib.load(model_file)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names')
            print(f"✅ Model loaded for {self.symbol}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise Exception("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions


def train_model_for_stock(symbol, force_refresh=False):
    """Complete pipeline: fetch data, engineer features, train model"""
    print(f"\n{'='*60}")
    print(f"🚀 Training pipeline for {symbol}")
    print(f"{'='*60}")
    
    # 1. Get data
    data_manager = StockDataManager()
    df = data_manager.fetch_and_cache_data(symbol, period='3y', force_refresh=force_refresh)
    
    if df is None or df.empty:
        print(f"❌ No data available for {symbol}")
        return False
    
    print(f"📊 Loaded {len(df)} rows of data")
    
    # 2. Add technical indicators
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.add_technical_indicators(df)
    
    if df_features is None:
        print(f"❌ Feature engineering failed for {symbol}")
        return False
    
    print(f"✅ Added technical indicators ({len(df_features.columns)} features)")
    
    # 3. Prepare features for training
    X_train, X_test, y_train, y_test, feature_names = feature_engineer.prepare_features(df_features, target_days=30)
    
    if X_train is None:
        print(f"❌ Feature preparation failed for {symbol}")
        return False
    
    print(f"✅ Prepared training data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # 4. Train model
    model = StockMLModel(symbol)
    model.feature_names = feature_names
    metrics = model.train(X_train, y_train, X_test, y_test)
    
    # 5. Save model
    model.save(metrics)
    
    print(f"✅ Training complete for {symbol}\n")
    return True


def predict_stock_price(symbol, days_ahead=30):
    """Make prediction using trained model"""
    # Load data
    data_manager = StockDataManager()
    df = data_manager.get_stock_data(symbol)
    
    if df is None:
        return None, "No data available"
    
    # Add features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.add_technical_indicators(df)
    
    if df_features is None:
        return None, "Feature engineering failed"
    
    # Load model (trained on 30-day target)
    model = StockMLModel(symbol)
    if not model.load():
        return None, "Model not trained. Please train first."
    
    # Get latest features
    if model.feature_names is None:
        # Use all numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        X_latest = df_features[numeric_cols].tail(1).values
    else:
        X_latest = df_features[model.feature_names].tail(1).values
    
    # Predict
    try:
        # Model is trained for 30 days, so we need to scale for different periods
        base_prediction = model.predict(X_latest)[0]
        current_price = df_features['Close'].iloc[-1]
        
        # Calculate the 30-day change
        base_change = base_prediction - current_price
        
        # Scale the change based on days_ahead (simple linear scaling)
        # This is approximate but better than using fixed 30-day prediction for all periods
        scaling_factor = days_ahead / 30.0
        
        # Apply scaling
        scaled_change = base_change * scaling_factor
        predicted_price = current_price + scaled_change
        
        # Calculate trend
        change_pct = (scaled_change / current_price) * 100
        
        trend = "Bullish 📈" if scaled_change > 0 else "Bearish 📉"
        
        # Confidence decreases with longer time horizons
        base_confidence = min(abs(change_pct) * 3, 85)
        confidence_penalty = max(0, (days_ahead - 30) / 60.0)  # Reduce confidence for longer periods
        confidence = max(base_confidence * (1 - confidence_penalty), 25)  # Minimum 25% confidence
        
        result = {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'change': float(scaled_change),
            'change_pct': float(change_pct),
            'trend': trend,
            'confidence': float(confidence),
            'days_ahead': days_ahead
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


if __name__ == "__main__":
    # Test the pipeline
    print("🧪 Testing stock ML pipeline...")
    
    # Train model for one stock as a test
    test_symbol = 'TCS.NS'
    success = train_model_for_stock(test_symbol, force_refresh=True)
    
    if success:
        print("\n🔮 Testing prediction...")
        result, error = predict_stock_price(test_symbol, days_ahead=30)
        
        if result:
            print(f"\n✅ Prediction successful!")
            print(f"Current Price: ₹{result['current_price']:.2f}")
            print(f"Predicted Price (30 days): ₹{result['predicted_price']:.2f}")
            print(f"Change: ₹{result['change']:.2f} ({result['change_pct']:.2f}%)")
            print(f"Trend: {result['trend']}")
            print(f"Confidence: {result['confidence']:.1f}%")
        else:
            print(f"❌ Prediction failed: {error}")

