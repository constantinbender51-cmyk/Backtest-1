import pandas as pd
import numpy as np
import json
import time
import os
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
        self.open_trades = []
        self.signals_processed = 0
        self.api_errors = 0
        self.checkpoint_interval = 1000
        self.checkpoint_file = "backtest_checkpoint.json"
        
    def load_data(self, filepath):
        """Load OHLC data from CSV"""
        print(f"Loading data from: {filepath}")
        
        # Read with space separator
        df = pd.read_csv(filepath, sep='\s+', engine='python')
        
        # Keep only needed columns and rename
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.rename(columns={'open_time': 'timestamp'})
        
        # Convert timestamp and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        print(f"Successfully loaded {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def prepare_ohlc_data(self, df, current_index):
        """Prepare OHLC data for API"""
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
    
    def run_backtest(self, df):
        """Run long-term backtest with checkpointing"""
        print("🚀 STARTING 7-DAY BACKTEST - TRADING EVERY SIGNAL!")
        print("⭐ This will take approximately 7 days to complete")
        print("💾 Checkpointing enabled - progress will be saved regularly")
        print("⏰ 24-hour time limit enabled for all trades")
        print("=" * 60)
        
        total_candles = len(df) - LOOKBACK_WINDOW
        start_time = time.time()
        start_index = LOOKBACK_WINDOW
        
        # Load checkpoint if exists
        if os.path.exists(self.checkpoint_file):
            start_index = self.load_checkpoint(df)
            print(f"📂 Resuming from checkpoint: Candle {start_index - LOOKBACK_WINDOW}/{total_candles}")
        
        for i in range(start_index, len(df)):
            current_row = df.iloc[i]
            current_price = current_row['close']
            timestamp = current_row.name
            candle_number = i - LOOKBACK_WINDOW + 1
            
            # Show progress
            if candle_number % 10 == 0:
                elapsed = time.time() - start_time
                progress_pct = (candle_number / total_candles) * 100
                eta_seconds = (elapsed / candle_number) * (total_candles - candle_number)
                eta_days = eta_seconds / 86400
                
                # Count time-limited exits
                time_exits = sum(1 for trade in self.trades if trade.get('exit_reason') == 'TIME_LIMIT_24H')
                
                print(f"📊 Candle {candle_number:,}/{total_candles:,} "
                      f"({progress_pct:.3f}%) - "
                      f"Elapsed: {elapsed/3600:.1f}h - "
                      f"ETA: {eta_days:.1f} days - "
                      f"API Calls: {self.signals_processed} - "
                      f"Trades: {len(self.trades)} - "
                      f"Time Exits: {time_exits}")
            
            # Check exits and execute new trade
            self.check_exit_conditions_all(current_price, timestamp)
            
            ohlc_data = self.prepare_ohlc_data(df, i)
            signal = self.deepseek_client.get_trading_signal(ohlc_data)
            self.signals_processed += 1
            
            self.execute_trade_parallel(signal, current_price, timestamp)
            
            # Save checkpoint every N candles
            if candle_number % self.checkpoint_interval == 0:
                self.save_checkpoint(i, df)
        
        # Final cleanup
        self.close_all_trades(df.iloc[-1]['close'], df.index[-1])
        self.save_final_results(start_time)
    
    def execute_trade_parallel(self, signal, current_price, timestamp):
        """Execute trade for every signal"""
        if signal['signal'] == 'BUY' and self.balance > 100:
            trade_amount = min(self.balance * TRADE_SIZE, self.balance)
            btc_amount = trade_amount / current_price
            
            trade_id = f"trade_{len(self.trades) + len(self.open_trades) + 1}"
            
            trade = {
                'id': trade_id,
                'type': 'LONG',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': float(signal['stop_price']) if signal['stop_price'] else current_price * 0.95,
                'target_price': float(signal['target_price']) if signal['target_price'] else current_price * 1.05,
                'size': trade_amount,
                'btc_amount': btc_amount,
                'signal_confidence': signal['confidence'],
                'signal_reason': signal.get('reason', '')
            }
            
            self.balance -= trade_amount * (1 + COMMISSION)
            self.open_trades.append(trade)
            
        elif signal['signal'] == 'SELL' and self.balance > 100:
            trade_amount = self.balance * TRADE_SIZE
            btc_amount = trade_amount / current_price
            
            trade_id = f"trade_{len(self.trades) + len(self.open_trades) + 1}"
            
            trade = {
                'id': trade_id,
                'type': 'SHORT',
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_price': float(signal['stop_price']) if signal['stop_price'] else current_price * 1.05,
                'target_price': float(signal['target_price']) if signal['target_price'] else current_price * 0.95,
                'size': trade_amount,
                'btc_amount': btc_amount,
                'signal_confidence': signal['confidence'],
                'signal_reason': signal.get('reason', '')
            }
            
            self.balance -= trade_amount * COMMISSION
            self.open_trades.append(trade)
    
    def check_exit_conditions_all(self, current_price, timestamp):
        """Check exit conditions for ALL open trades including time limit"""
        trades_to_remove = []
        
        for trade in self.open_trades:
            should_exit = False
            exit_reason = ""
            
            # Calculate trade duration in hours
            entry_time = trade['entry_time']
            trade_duration_hours = (timestamp - entry_time).total_seconds() / 3600
            
            # Time-based exit (24 hours)
            if trade_duration_hours >= 24:
                should_exit = True
                exit_reason = 'TIME_LIMIT_24H'
            
            # Price-based exits
            elif trade['type'] == 'LONG':
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
                # Add duration to trade info before closing
                trade['duration_hours'] = trade_duration_hours
                self.exit_trade_parallel(trade, current_price, timestamp, exit_reason)
                trades_to_remove.append(trade)
        
        # Remove closed trades from open trades list
        for trade in trades_to_remove:
            self.open_trades.remove(trade)
    
    def exit_trade_parallel(self, trade, exit_price, timestamp, exit_reason):
        """Exit a parallel trade"""
        # Calculate final duration
        entry_time = trade['entry_time']
        duration_hours = (timestamp - entry_time).total_seconds() / 3600
        
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
            'entry_time': entry_time.isoformat(),
            'exit_time': timestamp.isoformat(),
            'type': trade['type'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'size': trade['size'],
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'duration_hours': duration_hours,
            'signal_confidence': trade.get('signal_confidence', 0),
            'signal_reason': trade.get('signal_reason', '')
        }
        
        self.trades.append(trade_record)
        
        # Only print if it's a time limit exit to reduce noise
        if exit_reason == 'TIME_LIMIT_24H':
            print(f"⏰ Trade CLOSED: {exit_reason} after {duration_hours:.1f}h")
            print(f"   📋 ID: {trade['id']}")
            print(f"   🔄 Type: {trade['type']}")
            print(f"   📈 Entry: ${trade['entry_price']:.2f}")
            print(f"   📉 Exit: ${exit_price:.2f}")
            print(f"   💰 PnL: ${pnl:+.2f} ({pnl_percent:+.2f}%)")
            print("=" * 50)
    
    def close_all_trades(self, exit_price, timestamp):
        """Close all remaining trades"""
        for trade in self.open_trades[:]:
            # Add duration before closing
            entry_time = trade['entry_time']
            trade_duration_hours = (timestamp - entry_time).total_seconds() / 3600
            trade['duration_hours'] = trade_duration_hours
            self.exit_trade_parallel(trade, exit_price, timestamp, 'END_OF_DATA')
            self.open_trades.remove(trade)
    
    def save_checkpoint(self, current_index, df):
        """Save progress checkpoint"""
        checkpoint_data = {
            'current_index': current_index,
            'balance': self.balance,
            'signals_processed': self.signals_processed,
            'total_trades': len(self.trades),
            'open_trades_count': len(self.open_trades),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint saved at candle {current_index - LOOKBACK_WINDOW}")
    
    def load_checkpoint(self, df):
        """Load progress checkpoint"""
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            print(f"📂 Loaded checkpoint from {data['timestamp']}")
            self.balance = data['balance']
            self.signals_processed = data['signals_processed']
            
            return data['current_index']
        except:
            return LOOKBACK_WINDOW
    
    def save_final_results(self, start_time):
        """Save final results"""
        total_time = time.time() - start_time
        
        # Calculate time exit statistics
        time_exits = [t for t in self.trades if t['exit_reason'] == 'TIME_LIMIT_24H']
        stop_loss_exits = [t for t in self.trades if t['exit_reason'] == 'STOP_LOSS']
        target_exits = [t for t in self.trades if t['exit_reason'] == 'TARGET']
        end_exits = [t for t in self.trades if t['exit_reason'] == 'END_OF_DATA']
        
        results = {
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'total_time_days': total_time / 86400,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': self.balance - self.initial_balance,
            'return_percentage': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_signals': self.signals_processed,
            'total_trades': len(self.trades),
            'time_limit_exits': len(time_exits),
            'stop_loss_exits': len(stop_loss_exits),
            'target_exits': len(target_exits),
            'end_of_data_exits': len(end_exits),
            'api_call_rate': self.signals_processed / total_time if total_time > 0 else 0,
            'completion_time': datetime.now().isoformat()
        }
        
        with open('final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trades to CSV for analysis
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            trades_df.to_csv('all_trades.csv', index=False)
        
        print("=" * 60)
        print("🏁 BACKTEST COMPLETED!")
        print("=" * 60)
        print(f"⏰ Total time: {total_time/86400:.2f} days")
        print(f"📞 API calls: {self.signals_processed:,}")
        print(f"💰 Final balance: ${self.balance:,.2f}")
        print(f"📈 PnL: ${self.balance - self.initial_balance:,.2f}")
        print(f"📊 Return: {(self.balance - self.initial_balance)/self.initial_balance*100:.2f}%")
        print(f"🔁 Trades executed: {len(self.trades):,}")
        print(f"⏰ Time limit exits: {len(time_exits):,}")
        print(f"⛔ Stop loss exits: {len(stop_loss_exits):,}")
        print(f"🎯 Target exits: {len(target_exits):,}")
        print("💾 Results saved to final_results.json and all_trades.csv")
