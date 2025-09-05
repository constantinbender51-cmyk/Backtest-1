import requests
import json
import time
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MAX_REQUESTS_PER_SECOND, RETRY_DELAY, MAX_RETRIES

class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.last_call_time = 0
        self.min_interval = 1.0 / MAX_REQUESTS_PER_SECOND
    
    def get_trading_signal(self, ohlc_data):
        """Get trading signal from DeepSeek API with aggressive rate limiting"""
        
        # Rate limiting - ensure we don't exceed the rate limit
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            time_to_wait = self.min_interval - elapsed
            time.sleep(time_to_wait)
        
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
            "max_tokens": 500,
            "stream": False  # Ensure non-streaming for faster response
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                self.last_call_time = time.time()
                response = requests.post(
                    DEEPSEEK_API_URL, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=5  # Shorter timeout for faster retries
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = content[json_start:json_end]
                signal_data = json.loads(json_str)
                
                # Validate the signal
                if signal_data.get('signal') not in ['BUY', 'SELL', 'HOLD']:
                    raise ValueError("Invalid signal type")
                
                return signal_data
                
            except requests.exceptions.RequestException as e:
                print(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Response parsing failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                continue
        
        # If all retries failed, return HOLD signal
        print("All API attempts failed, returning HOLD signal")
        return {
            "signal": "HOLD",
            "stop_price": None,
            "target_price": None,
            "confidence": 0,
            "reason": "API Error - all retries failed"
        }
