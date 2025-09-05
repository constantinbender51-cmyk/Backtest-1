import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your-deepseek-api-key-here')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Trading Configuration
INITIAL_BALANCE = 10000  # USDT
TRADE_SIZE = 0.1  # 10% of portfolio per trade
COMMISSION = 0.001  # 0.1% trading fee

# Backtest Configuration
LOOKBACK_WINDOW = 100  # Number of candles to show to DeepSeek
