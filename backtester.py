import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from deepseek_client import DeepSeekClient
from config import INITIAL_BALANCE, TRADE_SIZE, COMMISSION, LOOKBACK_WINDOW

class Backtester:
    def __init__(self, deepseek_client):
        self.deepseek_client = deepseek_client
        self.initial_balance = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.btc_balance = 0
        self.trades = []
        self.current_trade = None
        self.signals_processed = 0
        self.api_errors = 0
        
    def load_data(self, filepath):
    """Load Binance-formatted CSV data with tab separators"""
    print(f"Loading data from: {filepath}")
    
    try:
        # Read CSV with tab separator
        df = pd.read_csv(filepath, sep='\t')
        print(f"Raw columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")
        
        # Display first few rows for verification
        print("First 3 rows:")
        print(df.head(3))
        
        # Keep only the columns we need and rename timestamp
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.rename(columns={'open_time': 'timestamp'})
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        print(f"Successfully loaded {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Data types:\n{df.dtypes}")
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Trying alternative loading methods...")
        
        # Fallback: try different separators
        try:
            df = pd.read_csv(filepath, sep=',')
            print(f"Comma separator worked. Columns: {df.columns.tolist()}")
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
            df = df.rename(columns={'open_time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except:
            try:
                df = pd.read_csv(filepath, sep=None, engine='python')
                print(f"Auto separator worked. Columns: {df.columns.tolist()}")
                df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
                df = df.rename(columns={'open_time': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e2:
                raise Exception(f"All loading methods failed: {e2}")
    
    def prepare_ohlc_data(self, df, current_index):
        """Prepare OHLC data for the current window - optimized version"""
        start_idx = max(0, current_index - LOOKBACK_WINDOW)
        window_data = df.iloc[start_idx:current_index + 1]
        
        # Use list comprehension for faster processing
        ohlc_data = [{
            "timestamp": row.name.isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row['volume'])
        } for _, row in window_data.iterrows()]
        
        return ohlc_data
    
    def execute_trade(self, signal, current_price, timestamp):
        """Execute a trade based on the signal"""
        if signal['signal'] == 'BUY' and self.current_trade is None:
            # LONG position
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            self.balance -= trade_amount * (1 + COMMISSION)
            self.btc_balance += btc_amount
            
            self.current_trade = {
                'type': 'LONG',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': signal['stop_price'],
                'target_price': signal['target_price'],
                'size': trade_amount
            }
            
        elif signal['signal'] == 'SELL' and self.current_trade is None:
            # SHORT position
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            self.balance -= trade_amount * COMMISSION
            self.btc_balance -= btc_amount
            
            self.current_trade = {
                'type': 'SHORT',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': signal['stop_price'],
                'target_price': signal['target_price'],
                'size': trade_amount
            }
    
    def check_exit_conditions(self, current_price, timestamp):
        """Check if current trade should be exited"""
        if self.current_trade is None:
            return False
        
        trade = self.current_trade
        
        if trade['type'] == 'LONG':
            if current_price <= trade['stop_price']:
                self.exit_trade(current_price, timestamp, 'STOP_LOSS')
                return True
            if current_price >= trade['target_price']:
                self.exit_trade(current_price, timestamp, 'TARGET')
                return True
                
        elif trade['type'] == 'SHORT':
            if current_price >= trade['stop_price']:
                self.exit_trade(current_price, timestamp, 'STOP_LOSS')
                return True
            if current_price <= trade['target_price']:
                self.exit_trade(current_price, timestamp, 'TARGET')
                return True
        
        return False
    
    def exit_trade(self, exit_price, timestamp, exit_reason):
        """Exit current trade"""
        if self.current_trade is None:
            return
        
        trade = self.current_trade
        
        if trade['type'] == 'LONG':
            exit_value = self.btc_balance * exit_price
            pnl = exit_value - trade['size']
            self.balance += exit_value * (1 - COMMISSION)
            self.btc_balance = 0
            
        elif trade['type'] == 'SHORT':
            buyback_cost = abs(self.btc_balance) * exit_price
            pnl = trade['size'] - buyback_cost
            self.balance += trade['size'] - buyback_cost
            self.btc_balance = 0
        
        self.trades.append({
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'type': trade['type'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'size': trade['size'],
            'pnl': pnl,
            'exit_reason': exit_reason
        })
        
        self.current_trade = None
    
    def run_backtest(self, df):
        """Run the complete backtest with optimized API calls"""
        print("Starting backtest...")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        print(f"Total candles to process: {len(df) - LOOKBACK_WINDOW}")
        
        start_time = time.time()
        
        for i in tqdm(range(LOOKBACK_WINDOW, len(df)), desc="Processing candles"):
            current_row = df.iloc[i]
            current_price = current_row['close']
            timestamp = current_row.name
            
            if self.check_exit_conditions(current_price, timestamp):
                continue
            
            if self.current_trade is None:
                ohlc_data = self.prepare_ohlc_data(df, i)
                signal = self.deepseek_client.get_trading_signal(ohlc_data)
                self.signals_processed += 1
                
                if signal.get('reason') == 'API Error - all retries failed':
                    self.api_errors += 1
                
                self.execute_trade(signal, current_price, timestamp)
        
        if self.current_trade is not None:
            self.exit_trade(df.iloc[-1]['close'], df.index[-1], 'END_OF_DATA')
        
        end_time = time.time()
        total_time = end_time - start_time
        candles_per_second = (len(df) - LOOKBACK_WINDOW) / total_time
        
        print(f"\nBacktest completed in {total_time:.2f} seconds")
        print(f"Processing rate: {candles_per_second:.2f} candles/second")
        print(f"Signals processed: {self.signals_processed}")
        print(f"API errors: {self.api_errors}")
    
    def generate_stats(self):
        """Generate performance statistics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_pnl = trades_df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        stats = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_usdt': total_pnl,
            'total_return_percent': total_return,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'signals_processed': self.signals_processed,
            'api_errors': self.api_errors
        }
        
        return stats
    
    def print_stats(self, stats):
        """Print formatted statistics"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Balance: ${stats['initial_balance']:,.2f}")
        print(f"Final Balance: ${stats['final_balance']:,.2f}")
        print(f"Total Return: ${stats['total_return_usdt']:,.2f} ({stats['total_return_percent']:.2f}%)")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Signals Processed: {stats['signals_processed']}")
        print(f"API Errors: {stats['api_errors']}")

# Add missing import
import time
