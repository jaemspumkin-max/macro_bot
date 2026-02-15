# 🚀 Macro Bias Engine - Quick Start Guide

Welcome! This guide will get you up and running with the Macro Bias Engine in 5 minutes.

## 📦 What's Included

```
📁 macro-bias-engine/
├── 📄 macro_bias_engine.py      # Main engine (core logic)
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Full documentation
├── 📄 usage_examples.py         # 7 comprehensive examples
├── 📄 flask_api.py              # REST API for web integration
├── 📄 dashboard.html            # Beautiful web dashboard
└── 📄 QUICKSTART.md             # This file
```

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
# Install core packages
pip install pandas numpy yfinance pandas-datareader

# Optional: For web dashboard
pip install flask flask-cors
```

### Step 2: Test the Engine (2 minutes)

Create a file called `test.py`:

```python
from macro_bias_engine import MacroBiasEngine, OutputFormatter

# Initialize and run
engine = MacroBiasEngine()
results = engine.run_analysis()

# Display results
OutputFormatter.print_summary(results)
```

Run it:
```bash
python test.py
```

You should see output like:
```
======================================================================
MACRO BIAS ENGINE - RESULTS SUMMARY
======================================================================

📊 OVERALL BIAS: Bullish
   Strength: +23.45% (Bullish)
   Confidence: 67.89%

🌊 MARKET REGIME: Medium Volatility
   Volatility: 45.23%
```

### Step 3: Run Examples (2 minutes)

```bash
python usage_examples.py
```

This runs 7 comprehensive examples showing different use cases.

## 🎯 Common Use Cases

### Use Case 1: Quick Market Check

**Goal**: Get a quick read on macro conditions

```python
from macro_bias_engine import MacroBiasEngine

engine = MacroBiasEngine()
results = engine.run_analysis()

# Quick access
bias = results['summary']['overall_bias']
strength = results['summary']['bias_strength_pct']
confidence = results['summary']['bias_confidence_pct']

print(f"Market is {bias} with {confidence:.1f}% confidence")
```

**Output**: `Market is Bullish with 67.9% confidence`

### Use Case 2: Custom Strategy

**Goal**: Emphasize specific factors for your trading style

```python
from macro_bias_engine import MacroBiasEngine

# Customize weights for bond trading
weights = {
    '10Y_Treasury_Yield': 3.0,      # High priority
    'Credit_Spreads_BAA_AAA': 3.0,  # High priority
    'DXY_Dollar_Index': 1.5,        # Medium priority
    'M2_Money_Supply': 1.0,         # Lower priority
    'VIX_Index': 2.0,               # Medium-high priority
}

engine = MacroBiasEngine(weights=weights)
results = engine.run_analysis()
```

### Use Case 3: Web Dashboard

**Goal**: Display results in a web interface

**Step 1**: Start the API server
```bash
python flask_api.py
```

**Step 2**: Open the dashboard
```bash
# Open dashboard.html in your browser
open dashboard.html
# or
start dashboard.html
# or
xdg-open dashboard.html
```

The dashboard will fetch data from the API and display it beautifully.

### Use Case 4: Automated Signals

**Goal**: Generate trading signals automatically

```python
from macro_bias_engine import MacroBiasEngine

engine = MacroBiasEngine()
results = engine.run_analysis()

# Extract key metrics
bias_strength = results['summary']['bias_strength_pct']
confidence = results['summary']['bias_confidence_pct']

# Generate signal
if bias_strength > 30 and confidence > 60:
    signal = "STRONG BUY"
elif bias_strength > 15 and confidence > 40:
    signal = "BUY"
elif bias_strength < -30 and confidence > 60:
    signal = "STRONG SELL"
elif bias_strength < -15 and confidence > 40:
    signal = "SELL"
else:
    signal = "NEUTRAL"

print(f"Trading Signal: {signal}")
```

### Use Case 5: Export for Analysis

**Goal**: Save results for later analysis or backtesting

```python
from macro_bias_engine import MacroBiasEngine, OutputFormatter

engine = MacroBiasEngine()
results = engine.run_analysis()

# Export to JSON
json_data = OutputFormatter.to_json(results)
with open('bias_results.json', 'w') as f:
    f.write(json_data)

# Export to CSV
df = OutputFormatter.to_dataframe(results)
df.to_csv('factor_scores.csv', index=False)

print("✅ Results saved!")
```

## 🔧 Customization Cheat Sheet

### Adjust Factor Weights

```python
weights = {
    '10Y_Treasury_Yield': 2.0,      # Default: 2.0
    'DXY_Dollar_Index': 2.0,        # Default: 2.0
    'M2_Money_Supply': 2.0,         # Default: 2.0
    'Credit_Spreads_BAA_AAA': 1.5,  # Default: 1.5
    'VIX_Index': 1.5,               # Default: 1.5
    'Economic_Surprises': 1.0       # Default: 1.0
}
```

**Tips**:
- Increase weight (3.0+) for factors you trust most
- Decrease weight (0.5-1.0) for factors you trust less
- Total weight doesn't matter, only relative weights

### Change Data Lookback Period

```python
from macro_bias_engine import MacroDataFetcher

# Fetch 180 days instead of default 90
fetcher = MacroDataFetcher(lookback_days=180)
```

### Adjust Volatility Thresholds

Edit these in `macro_bias_engine.py`:

```python
VOLATILITY_THRESHOLDS = {
    'Low': (0, 33),      # 0-33% = Low volatility
    'Medium': (34, 66),  # 34-66% = Medium
    'High': (67, 100)    # 67-100% = High
}
```

## 📊 Understanding the Output

### Overall Bias
- **Bullish**: Market conditions favor upside
- **Bearish**: Market conditions favor downside
- **Neutral**: Mixed signals, no clear direction

### Bias Strength (-100 to +100%)
- **+50 to +100**: Very bullish
- **+15 to +50**: Moderately bullish
- **-15 to +15**: Neutral zone
- **-50 to -15**: Moderately bearish
- **-100 to -50**: Very bearish

### Confidence (0 to 100%)
- **80-100%**: Very high confidence (strong signal)
- **60-80%**: High confidence
- **40-60%**: Moderate confidence
- **20-40%**: Low confidence
- **0-20%**: Very low confidence (noisy)

### Market Regime
- **Low Volatility**: Calm markets, trending conditions
- **Medium Volatility**: Normal market conditions
- **High Volatility**: Stressed markets, higher risk

## 🚨 Troubleshooting

### "ModuleNotFoundError: No module named 'yfinance'"

**Solution**:
```bash
pip install yfinance pandas-datareader
```

### "Unable to fetch data from FRED"

**Possible causes**:
1. No internet connection
2. FRED API is down (rare)
3. Series has been discontinued

**Solution**: Wait a few minutes and try again. FRED is usually very reliable.

### "Yahoo Finance returns empty data"

**Possible causes**:
1. Market is closed (for VIX, DXY)
2. Yahoo Finance API issues

**Solution**: Try again during market hours or wait 15 minutes.

### API not accessible from dashboard

**Solution**:
1. Make sure Flask API is running: `python flask_api.py`
2. Check that port 5000 is not blocked
3. Open dashboard.html in a browser (not just viewing the file)

## 🎓 Next Steps

### Beginner
1. ✅ Run `python usage_examples.py`
2. ✅ Understand each factor's meaning (see README.md)
3. ✅ Try custom weights for your strategy

### Intermediate
1. ✅ Set up the Flask API
2. ✅ Build custom dashboard or integrate with existing tools
3. ✅ Backtest signals against historical data

### Advanced
1. ✅ Add machine learning predictions
2. ✅ Integrate with portfolio management system
3. ✅ Build automated trading signals
4. ✅ Create alerts for regime changes

## 📚 Additional Resources

### Documentation
- **README.md**: Comprehensive documentation
- **usage_examples.py**: 7 detailed examples
- **macro_bias_engine.py**: Heavily commented source code

### Integration Examples
- **flask_api.py**: REST API for web apps
- **dashboard.html**: Beautiful web interface
- Comments in code show React/Vue/Angular integration

### Data Sources
- FRED (Federal Reserve): https://fred.stlouisfed.org
- Yahoo Finance: https://finance.yahoo.com
- Both are free and don't require API keys

## 💡 Pro Tips

### Tip 1: Cache Your Results
The engine fetches live data every time. For production, implement caching:

```python
import pickle
from datetime import datetime, timedelta

def get_cached_results():
    try:
        with open('cache.pkl', 'rb') as f:
            cached = pickle.load(f)
            if datetime.now() - cached['timestamp'] < timedelta(hours=4):
                return cached['results']
    except:
        pass
    
    # Fetch new data
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    
    # Save to cache
    with open('cache.pkl', 'wb') as f:
        pickle.dump({'timestamp': datetime.now(), 'results': results}, f)
    
    return results
```

### Tip 2: Combine with Technical Analysis
Don't use macro bias alone. Combine it with:
- Price action and trends
- Support/resistance levels
- RSI, MACD, moving averages
- Volume analysis

### Tip 3: Track Regime Changes
Log results over time to identify regime shifts:

```python
import json
from datetime import datetime

# Run analysis
engine = MacroBiasEngine()
results = engine.run_analysis()

# Append to history file
with open('bias_history.jsonl', 'a') as f:
    f.write(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'bias': results['summary']['overall_bias'],
        'strength': results['summary']['bias_strength_pct'],
        'confidence': results['summary']['bias_confidence_pct'],
        'regime': results['summary']['regime']
    }) + '\n')
```

### Tip 4: Position Sizing
Adjust position sizes based on confidence and volatility:

```python
confidence = results['summary']['bias_confidence_pct']
volatility = results['summary']['volatility_pct']

# Base position: 100%
base_size = 100

# Adjust for confidence
confidence_multiplier = confidence / 100

# Adjust for volatility (inverse relationship)
volatility_multiplier = (100 - volatility) / 100

# Final position size
position_size = base_size * confidence_multiplier * volatility_multiplier

print(f"Recommended position size: {position_size:.1f}%")
```

## 🤝 Community & Support

### Getting Help
1. Check the README.md for detailed documentation
2. Review usage_examples.py for code examples
3. Read comments in macro_bias_engine.py

### Contributing
Contributions welcome! Ideas for enhancements:
- Additional data sources (sentiment, positioning)
- Machine learning models
- Backtesting framework
- Mobile app integration

## ⚖️ Disclaimer

**IMPORTANT**: This tool is for informational and educational purposes only. It is NOT financial advice. 

- Always do your own research
- Consult with licensed financial advisors
- Past performance doesn't guarantee future results
- Markets are inherently unpredictable

The authors take no responsibility for trading losses or investment decisions made using this tool.

## 📧 Questions?

If you have questions or find bugs:
1. Review the documentation thoroughly
2. Check usage_examples.py for similar use cases
3. Ensure all dependencies are installed correctly

---

**Happy Trading! 🚀**

Remember: The best use of this engine is as ONE INPUT among many in your trading decisions. Combine it with fundamental analysis, technical analysis, risk management, and your own judgment.
