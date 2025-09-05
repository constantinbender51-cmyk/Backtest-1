import os
import time
from dotenv import load_dotenv
from backtester import Backtester
from deepseek_client import DeepSeekClient
from config import DEEPSEEK_API_KEY

def main():
    # Load environment variables
    load_dotenv()
    
    # Wait a bit for Railway to fully start
    time.sleep(2)
    
    # Check if API key is available
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == 'your-deepseek-api-key-here':
        print("❌ ERROR: DeepSeek API key not found!")
        print("Please set the DEEPSEEK_API_KEY environment variable in Railway")
        return
    
    # Initialize clients
    deepseek_client = DeepSeekClient(DEEPSEEK_API_KEY)
    backtester = Backtester(deepseek_client)
    
    # Load data - file should be in the same directory
    data_file = 'history.csv'
    
    try:
        df = backtester.load_data(data_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Current directory contents:", os.listdir('.'))
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
        
        # Show first few trades
        print("\nFirst few trades:")
        print(trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].head())

if __name__ == "__main__":
    main()
