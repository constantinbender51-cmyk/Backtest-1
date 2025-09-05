import pandas as pd
import numpy as np
import json
import time
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
        self.open_trades = []  # List of active trades for parallel trading
        self.signals_processed = 0
        self.api_errors = 0
        
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
    
    def run_backtest(self, df):
        """Run backtest with parallel trading - every signal is traded"""
        print("🚀 Starting parallel backtest - trading every signal!")
        print(f"📊 Data range: {df.index[0]} to {df.index[-1]}")
        total_candles = len(df) - LOOKBACK_WINDOW
        print(f"🕯️  Total candles to process: {total_candles}")
        print("=" * 60)
        
        start_time = time.time()
        api_call_count = 0
        trade_count = 0
        last_update_time = time.time()
        
        # Process each candle
        for i in range(LOOKBACK_WINDOW, len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            timestamp = current_row.name
            candle_number = i - LOOKBACK_WINDOW + 1
            
            # Show progress every 100 candles or every 30 seconds
            current_time = time.time()
            if candle_number % 100 == 0 or current_time - last_update_time >= 30:
                elapsed = current_time - start_time
                candles_per_second = candle_number / elapsed if elapsed > 0 else 0
                estimated_total = elapsed / candle_number * total_candles if candle_number > 0 else 0
                remaining = estimated_total - elapsed
                
                print(f"📊 Candle {candle_number}/{total_candles} "
                      f"({candle_number/total_candles*100:.1f}%) - "
                      f"Elapsed: {elapsed:.0f}s - "
                      f"ETA: {remaining:.0f}s - "
                      f"Speed: {candles_per_second:.2f} candles/s - "
                      f"API Calls: {api_call_count} - "
                      f"Open Trades: {len(self.open_trades)} - "
                      f"Closed Trades: {len(self.trades)}")
                last_update_time = current_time
            
            # Check exit conditions for ALL open trades first
            self.check_exit_conditions_all(current_price, timestamp)
            
            # Get new signal for EVERY candle (regardless of open positions)
            ohlc_data = self.prepare_ohlc_data(df, i)
            signal = self.deepseek_client.get_trading_signal(ohlc_data)
            api_call_count += 1
            
            # Show API call progress every 10 calls
            if api_call_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"📡 API Call {api_call_count}: {signal['signal']} signal "
                      f"(Candle {candle_number}, {elapsed:.0f}s elapsed)")
            
            # Execute trade for EVERY signal (parallel trading)
            self.execute_trade_parallel(signal, current_price, timestamp)
            trade_count += 1
        
        # Close all remaining open trades at the end
        self.close_all_trades(df.iloc[-1]['close'], df.index[-1])
        
        end_time = time.time()
        total_time = end_time - start_time
        hours = total_time / 3600
        minutes = total_time / 60
        
        print("=" * 60)
        print("🏁 Parallel Backtest Completed!")
        print("=" * 60)
        print(f"⏰ Total time: {total_time:.0f} seconds ({minutes:.1f} minutes, {hours:.2f} hours)")
        print(f"📞 API calls made: {api_call_count}")
        print(f"📊 Candles processed: {total_candles}")
        print(f"💰 Signals generated: {trade_count}")
        print(f"📈 Trades executed: {len(self.trades)}")
        print(f"⚡ Processing speed: {total_candles/total_time:.2f} candles/second")
        print(f"📊 API call rate: {api_call_count/total_time:.2f} calls/second")
        print(f"💵 Final balance: ${self.balance:,.2f}")
        print(f"💰 PnL: ${self.balance - self.initial_balance:,.2f}")
        print(f"📈 Return: {(self.balance - self.initial_balance)/self.initial_balance*100:.2f}%")
    
    def execute_trade_parallel(self, signal, current_price, timestamp):
        """Execute trade for every signal (parallel trading)"""
        if signal['signal'] == 'BUY':
            trade_amount = self.balance * TRADE_SIZE
            if trade_amount > self.balance:
                print("❌ Insufficient balance for BUY")
                return
                
            btc_amount = trade_amount / current_price
            
            trade_id = f"trade_{len(self.trades) + len(self.open_trades) + 1}_{timestamp.strftime('%Y%m%d_%H%M')}"
            
            trade = {
                'id': trade_id,
                'type': 'LONG',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': signal['stop_price'],
                'target_price': signal['target_price'],
                'size': trade_amount,
                'btc_amount': btc_amount,
                'signal_confidence': signal['confidence'],
                'signal_reason': signal['reason']
            }
            
            self.balance -= trade_amount * (1 + COMMISSION)
            self.open_trades.append(trade)
            
            print(f"🎯 LONG executed at ${current_price:.2f}")
            print(f"   📋 ID: {trade_id}")
            print(f"   ⛔ Stop: ${signal['stop_price']:.2f}")
            print(f"   🎯 Target: ${signal['target_price']:.2f}")
            print(f"   ✅ Confidence: {signal['confidence']}%")
            print(f"   💰 Size: ${trade_amount:,.2f}")
            print("-" * 50)
            
        elif signal['signal'] == 'SELL':
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            trade_id = f"trade_{len(self.trades) + len(self.open_trades) + 1}_{timestamp.strftime('%Y%m%d_%H%M')}"
            
            trade = {
                'id': trade_id,
                'type': 'SHORT',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': signal['stop_price'],
                'target_price': signal['target_price'],
                'size': trade_amount,
                'btc_amount': btc_amount,
                'signal_confidence': signal['confidence'],
                'signal_reason': signal['reason']
            }
            
            self.balance -= trade_amount * COMMISSION
            self.open_trades.append(trade)
            
            print(f"🎯 SHORT executed at ${current_price:.2f}")
            print(f"   📋 ID: {trade_id}")
            print(f"   ⛔ Stop: ${signal['stop_price']:.2f}")
            print(f"   🎯 Target: ${signal['target_price']:.2f}")
            print(f"   ✅ Confidence: {signal['confidence']}%")
            print(f"   💰 Size: ${trade_amount:,.2f}")
            print("-" * 50)
    
    def check_exit_conditions_all(self, current_price, timestamp):
        """Check exit conditions for ALL open trades"""
        trades_to_remove = []
        
        for trade in self.open_trades:
            should_exit = False
            exit_reason = ""
            
            if trade['type'] == 'LONG':
                if current_price <= trade['stop_price']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                elif current_price >= trade['target_price']:
                    should_exit = True
                    exit_reason = 'TARGET'
                    
            elif trade['type'] == 'SHORT':
                if current_price >= trade['stop_price']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                elif current_price <= trade['target_price']:
                    should_exit = True
                    exit_reason = 'TARGET'
            
            if should_exit:
                self.exit_trade_parallel(trade, current_price, timestamp, exit_reason)
                trades_to_remove.append(trade)
        
        # Remove closed trades from open trades list
        for trade in trades_to_remove:
            self.open_trades.remove(trade)
    
    def exit_trade_parallel(self, trade, exit_price, timestamp, exit_reason):
        """Exit a parallel trade"""
        if trade['type'] == 'LONG':
            exit_value = trade['btc_amount'] * exit_price
            pnl = exit_value - trade['size']
            pnl_percent = (pnl / trade['size']) * 100
            
            self.balance += exit_value * (1 - COMMISSION)
            
        elif trade['type'] == 'SHORT':
            buyback_cost = trade['btc_amount'] * exit_price
            pnl = trade['size'] - buyback_cost
            pnl_percent = (pnl / trade['size']) * 100
            
            self.balance += trade['size'] - buyback_cost
        
        # Record trade
        trade_record = {
            'id': trade['id'],
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'type': trade['type'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'size': trade['size'],
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'duration': (timestamp - trade['entry_time']).total_seconds() / 3600,
            'signal_confidence': trade.get('signal_confidence', 0),
            'signal_reason': trade.get('signal_reason', '')
        }
        
        self.trades.append(trade_record)
        
        print(f"📊 Trade CLOSED: {exit_reason}")
        print(f"   📋 ID: {trade['id']}")
        print(f"   🔄 Type: {trade['type']}")
        print(f"   📈 Entry: ${trade['entry_price']:.2f}")
        print(f"   📉 Exit: ${exit_price:.2f}")
        print(f"   💰 PnL: ${pnl:+.2f} ({pnl_percent:+.2f}%)")
        print(f"   ⏱️  Duration: {trade_record['duration']:.1f}h")
        print(f"   ✅ Signal Confidence: {trade.get('signal_confidence', 0)}%")
        print("=" * 50)
    
    def close_all_trades(self, exit_price, timestamp):
        """Close all remaining open trades at end of backtest"""
        print(f"🔚 Closing {len(self.open_trades)} remaining trades...")
        for trade in self.open_trades[:]:  # Copy list for iteration
            self.exit_trade_parallel(trade, exit_price, timestamp, 'END_OF_DATA')
            self.open_trades.remove(trade)
    
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
