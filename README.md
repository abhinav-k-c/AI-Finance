# 🤖 AI-Powered Financial Advisor

A comprehensive personal finance management application with AI-powered insights, real-time market data, and intelligent recommendations.

## 🚀 Features

- **AI Financial Advisor Chat**: Get personalized financial advice using Google Gemini AI
- **Multi-Language Support**: Available in English, Hindi, Tamil, Telugu, and more
- **Data Management**: Manual entry and CSV upload for transactions
- **Financial Analysis**: Smart alerts, spending patterns, and visual insights
- **Goal Management**: Set, track, and achieve your financial goals
- **Investment Insights**: Real-time market data, stock predictions, portfolio tracking
- **Stock Predictions**: ML-based price predictions with technical indicators

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhinav-k-c/AI-Finance.git
   cd AI-Finance
   
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate on macOS/Linux:
   source venv/bin/activate
   
   # Activate on Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

Run the Streamlit application:
```bash
streamlit run Homepage.py
```

The application will open in your default browser at `http://localhost:8501`

## 📱 Usage

1. **Start on Homepage**: Get an overview of all features
2. **Add Data**: Go to "Data Input & Management" to add your financial data
3. **Analyze**: View insights in "Financial Analysis & Insights"
4. **Set Goals**: Create and track goals in "Goal Management"
5. **Chat with AI**: Get personalized advice in "AI Financial Advisor Chat"
6. **Invest Smart**: Analyze stocks and get recommendations in "Investment Insights"

## 🔑 API Keys

For AI Chat functionality, you'll need a Google Gemini API key:
- Get your key from: https://makersuite.google.com/app/apikey
- Replace the API key 

## 📊 Sample Data Format

CSV files should have these columns:
- `date`: Date of transaction (YYYY-MM-DD)
- `description`: Transaction description
- `amount`: Amount (numeric)
- `category`: Category name
- `type`: 'income' or 'expense'

## 🎯 Key Technologies

- **Frontend**: Streamlit
- **AI**: Google Gemini API
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly
- **Market Data**: Yahoo Finance API
- **ML Predictions**: scikit-learn
- **Database**: SQLite

## 📝 License

This project is for educational and personal use.

## 🤝 Support

For issues or questions, please refer to the documentation in each module.






