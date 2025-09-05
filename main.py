import os
from dotenv import load_dotenv
from backtester import Backtester
from deepseek_client import DeepSeekClient
from config import DEEPSEEK_API_KEY

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    deepseek_client = DeepSeekClient(DEEPSEEK_API_KEY)
    backtester = Backtester(deepseek_client)
    
    # Load data
    try:
        print("Loading historical data...")
        df = backtester.load_data('history.csv')
        print(f"Loaded {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    except FileNotFoundError:
        print("Error: history.csv not found")
        print("Please ensure the file exists with columns: timestamp,open,high,low,close,volume")
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

if __name__ == "__main__":
    main()
