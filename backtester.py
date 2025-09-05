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
        
    def load_data(self, filepath):
        """Load OHLC data from CSV with Binance format"""
        print(f"Loading data from: {filepath}")
        
        # Load the CSV with the correct column names
        df = pd.read_csv(filepath, sep='\t')  # Use tab separator if that's what your file uses
        
        # If tab separator doesn't work, try comma
        if len(df.columns) == 1:
            df = pd.read_csv(filepath)
        
        print(f"Columns found: {df.columns.tolist()}")
        
        # Rename columns to match expected format
        column_mapping = {
            'open_time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Keep only the columns we need and rename them
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.rename(columns=column_mapping)
        
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
        print(f"Data types:\n{df.dtypes}")
        
        return df
    
    def prepare_ohlc_data(self, df, current_index):
        """Prepare OHLC data for the current window"""
        start_idx = max(0, current_index - LOOKBACK_WINDOW)
        window_data = df.iloc[start_idx:current_index + 1]
        
        ohlc_data = []
        for _, row in window_data.iterrows():
            ohlc_data.append({
                "timestamp": row.name.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return ohlc_data
    
    def execute_trade(self, signal, current_price, timestamp):
        """Execute a trade based on the signal"""
        if signal['signal'] == 'BUY' and self.current_trade is None:
            # LONG position
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            # Update balances
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
            
            print(f"📈 BUY signal executed at ${current_price:.2f}")
            print(f"   Stop: ${signal['stop_price']:.2f}, Target: ${signal['target_price']:.2f}")
            print(f"   Confidence: {signal['confidence']}%, Reason: {signal['reason']}")
            
        elif signal['signal'] == 'SELL' and self.current_trade is None:
            # SHORT position
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            # For short selling, we borrow BTC and sell it
            # We'll track the borrowed amount as negative BTC balance
            self.balance -= trade_amount * COMMISSION  # Only pay commission
            self.btc_balance -= btc_amount  # Negative balance indicates short position
            
            self.current_trade = {
                'type': 'SHORT',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': signal['stop_price'],
                'target_price': signal['target_price'],
                'size': trade_amount
            }
            
            print(f"📉 SELL signal executed at ${current_price:.2f}")
            print(f"   Stop: ${signal['stop_price']:.2f}, Target: ${signal['target_price']:.2f}")
            print(f"   Confidence: {signal['confidence']}%, Reason: {signal['reason']}")
    
    def check_exit_conditions(self, current_price, timestamp):
        """Check if current trade should be exited"""
        if self.current_trade is None:
            return False
        
        trade = self.current_trade
        
        if trade['type'] == 'LONG':
            # Check stop loss (price goes down)
            if current_price <= trade['stop_price']:
                self.exit_trade(current_price, timestamp, 'STOP_LOSS')
                return True
            
            # Check target (price goes up)
            if current_price >= trade['target_price']:
                self.exit_trade(current_price, timestamp, 'TARGET')
                return True
                
        elif trade['type'] == 'SHORT':
            # Check stop loss (price goes up)
            if current_price >= trade['stop_price']:
                self.exit_trade(current_price, timestamp, 'STOP_LOSS')
                return True
            
            # Check target (price goes down)
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
            # Calculate PnL for long position
            exit_value = self.btc_balance * exit_price
            pnl = exit_value - trade['size']
            pnl_percent = (pnl / trade['size']) * 100
            
            # Update balances
            self.balance += exit_value * (1 - COMMISSION)
            self.btc_balance = 0
            
        elif trade['type'] == 'SHORT':
            # Calculate PnL for short position
            # We need to buy back the borrowed BTC at current price
            buyback_cost = abs(self.btc_balance) * exit_price
            pnl = trade['size'] - buyback_cost  # Profit if price went down
            pnl_percent = (pnl / trade['size']) * 100
            
            # Update balances - we return the borrowed BTC and keep the profit
            self.balance += trade['size'] - buyback_cost
            self.btc_balance = 0
        
        # Record trade
        trade_record = {
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'type': trade['type'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'size': trade['size'],
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'duration': (timestamp - trade['entry_time']).total_seconds() / 3600  # hours
        }
        
        self.trades.append(trade_record)
        
        print(f"📊 Trade closed: {exit_reason}")
        print(f"   Type: {trade['type']}, Entry: ${trade['entry_price']:.2f}, Exit: ${exit_price:.2f}")
        print(f"   PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        print(f"   Duration: {trade_record['duration']:.1f} hours")
        
        self.current_trade = None
    
    def run_backtest(self, df):
        """Run the complete backtest"""
        print("Starting backtest...")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        for i in tqdm(range(LOOKBACK_WINDOW, len(df)), desc="Processing candles"):
            current_row = df.iloc[i]
            current_price = current_row['close']
            timestamp = current_row.name
            
            # Check if we need to exit current trade
            if self.check_exit_conditions(current_price, timestamp):
                continue
            
            # If no active trade, get new signal
            if self.current_trade is None:
                ohlc_data = self.prepare_ohlc_data(df, i)
                signal = self.deepseek_client.get_trading_signal(ohlc_data)
                self.execute_trade(signal, current_price, timestamp)
        
        # Close any open trade at the end
        if self.current_trade is not None:
            self.exit_trade(df.iloc[-1]['close'], df.index[-1], 'END_OF_DATA')
    
    def generate_stats(self):
        """Generate performance statistics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Separate long and short trades
        long_trades = trades_df[trades_df['type'] == 'LONG']
        short_trades = trades_df[trades_df['type'] == 'SHORT']
        
        # Basic stats
        total_pnl = trades_df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Win rate
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Quarterly returns
        trades_df['quarter'] = trades_df['exit_time'].dt.to_period('Q')
        quarterly_returns = trades_df.groupby('quarter')['pnl'].sum()
        
        stats = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_usdt': total_pnl,
            'total_return_percent': total_return,
            'total_trades': len(trades_df),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_trade_duration_hours': trades_df['duration'].mean(),
            'long_win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
            'quarterly_returns': quarterly_returns.to_dict()
        }
        
        return stats
    
    def print_stats(self, stats):
        """Print formatted statistics"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Balance: ${stats['initial_balance']:,.2f}")
        print(f"Final Balance: ${stats['final_balance']:,.2f}")
        print(f"Total Return: ${stats['total_return_usdt']:,.2f} ({stats['total_return_percent']:.2f}%)")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Long Trades: {stats['long_trades']}")
        print(f"Short Trades: {stats['short_trades']}")
        print(f"Winning Trades: {stats['winning_trades']}")
        print(f"Losing Trades: {stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Long Win Rate: {stats['long_win_rate']:.1f}%")
        print(f"Short Win Rate: {stats['short_win_rate']:.1f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Avg Win: ${stats['avg_win']:,.2f}")
        print(f"Avg Loss: ${stats['avg_loss']:,.2f}")
        print(f"Largest Win: ${stats['largest_win']:,.2f}")
        print(f"Largest Loss: ${stats['largest_loss']:,.2f}")
        print(f"Avg Trade Duration: {stats['avg_trade_duration_hours']:.1f} hours")
        
        print("\nQuarterly Returns:")
        for quarter, returns in stats['quarterly_returns'].items():
            print(f"  {quarter}: ${returns:,.2f}")
        
        # Show trade type performance
        if stats['long_trades'] > 0 or stats['short_trades'] > 0:
            print(f"\nTrade Type Performance:")
            if stats['long_trades'] > 0:
                long_pnl = sum(trade['pnl'] for trade in self.trades if trade['type'] == 'LONG')
                print(f"  LONG: ${long_pnl:,.2f} ({stats['long_win_rate']:.1f}% win rate)")
            if stats['short_trades'] > 0:
                short_pnl = sum(trade['pnl'] for trade in self.trades if trade['type'] == 'SHORT')
                print(f"  SHORT: ${short_pnl:,.2f} ({stats['short_win_rate']:.1f}% win rate)")
