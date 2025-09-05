import os
import time
from dotenv import load_dotenv
from backtester import Backtester
from deepseek_client import DeepSeekClient
from config import DEEPSEEK_API_KEY

def main():
    # Load environment variables
    load_dotenv()
    
    # Wait a bit for Railway to fully start (optional)
    time.sleep(2)
    
    # Initialize clients
    deepseek_client = DeepSeekClient(DEEPSEEK_API_KEY)
    backtester = Backtester(deepseek_client)
    
    # Load data - check if file exists
    data_file = 'history.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please ensure the file exists in the root directory.")
        print("Current directory contents:", os.listdir('.'))
        return
    
    try:
        print("Loading historical data...")
        df = backtester.load_data(data_file)
        print(f"Loaded {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run backtest
    backtester.run_backtest(df)
    
    # Generate and print statistics
    stats = backtester.generate_stats()
    backtester.print_stats(stats)
    
    # Save trades to CSV
    if backtester.trades:
        import pandas as pd
        trades_df = pd.DataFrame(backtester.trades)
        trades_df.to_csv('trades_history.csv', index=False)
        print(f"\nTrades saved to trades_history.csv")
        
        # Also save to Railway's persistent storage if available
        try:
            trades_df.to_csv('/data/trades_history.csv', index=False)
            print("Trades also saved to persistent storage")
        except:
            pass

if __name__ == "__main__":
    main()
