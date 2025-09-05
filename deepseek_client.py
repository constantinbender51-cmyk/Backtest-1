import requests
import json
import time
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL

class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.last_call_time = 0
        self.min_interval = 1.0  # 1 second between calls
    
    def get_trading_signal(self, ohlc_data):
        """Get trading signal from DeepSeek API with simple rate limiting"""
        
        # Simple rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
        
        prompt = f"""Analyze the following BTC/USDT hourly OHLC data and generate a trading signal.
        Respond with ONLY a JSON object containing: signal (BUY|SELL|HOLD), stop_price, target_price, 
        confidence (0-100), and reason.

        OHLC Data (most recent last):
        {json.dumps(ohlc_data, indent=2)}

        Important: Consider technical analysis, price action, volume patterns, and market structure.
        Provide realistic stop and target prices based on support/resistance levels.
        Return ONLY JSON, no other text."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert cryptocurrency trading analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            return json.loads(json_str)
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return {
                "signal": "HOLD",
                "stop_price": None,
                "target_price": None,
                "confidence": 0,
                "reason": "API Error"
            }
