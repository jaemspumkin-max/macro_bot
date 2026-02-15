# Macro Bias Engine for Trading

A quantitative Python engine that evaluates multiple macroeconomic and market factors to determine directional bias, confidence, strength, volatility, and market regime for trading decisions.

## 🎯 Features

- **Multi-Factor Analysis**: Evaluates 6 key macroeconomic indicators
- **Weighted Scoring System**: Customizable weights for each factor
- **Real-Time Data**: Fetches live data from FRED and Yahoo Finance
- **Quantitative Output**: Bias strength, confidence, volatility, and regime classification
- **Dashboard Ready**: JSON, DataFrame, and dictionary outputs for easy integration
- **Production Quality**: Modular, well-documented, and robust code

## 📊 Factors Analyzed

| Factor | Weight | Bullish When | Bearish When | Data Source |
|--------|--------|--------------|--------------|-------------|
| 10-Year US Treasury Yield | 2.0 | Falling | Rising | FRED |
| US Dollar Index (DXY) | 2.0 | Falling | Rising | Yahoo Finance |
| M2 Money Supply | 2.0 | Rising | Falling | FRED |
| Credit Spreads (BAA-AAA) | 1.5 | Tightening | Widening | FRED |
| VIX Index | 1.5 | Falling | Rising | Yahoo Finance |
| Economic Surprises | 1.0 | Positive | Negative | Optional |

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install pandas numpy yfinance pandas-datareader

# Optional: For dashboard
pip install streamlit plotly

# Optional: For Flask API
pip install flask flask-cors
```

### Basic Usage

```python
from macro_bias_engine import MacroBiasEngine, OutputFormatter

# Initialize engine with default weights
engine = MacroBiasEngine()

# Run analysis
results = engine.run_analysis()

# Display results
OutputFormatter.print_summary(results)
```

### Output Example

```
======================================================================
MACRO BIAS ENGINE - RESULTS SUMMARY
======================================================================

📊 OVERALL BIAS: Bullish
   Strength: +23.45% (Bullish)
   Confidence: 67.89%

🌊 MARKET REGIME: Medium Volatility
   Volatility: 45.23%

📈 FACTOR BREAKDOWN:
----------------------------------------------------------------------
🟢 M2_Money_Supply                | Score: +0.78 | Weight: 2.0 | Contribution: 28.5%
🔴 10Y_Treasury_Yield             | Score: -0.45 | Weight: 2.0 | Contribution: 22.1%
🟢 DXY_Dollar_Index               | Score: +0.32 | Weight: 2.0 | Contribution: 19.8%
🔴 VIX_Index                      | Score: -0.23 | Weight: 1.5 | Contribution: 15.7%
🟢 Credit_Spreads_BAA_AAA         | Score: +0.18 | Weight: 1.5 | Contribution: 13.9%
```

## 📈 How It Works

### 1. Data Fetching

The engine fetches real-time macroeconomic data:
- **FRED API**: Treasury yields, M2 money supply, credit spreads
- **Yahoo Finance**: VIX, DXY dollar index

### 2. Factor Scoring

Each factor is normalized to a **-1 to +1 scale**:
- **Z-score calculation**: Standardizes current value vs historical distribution
- **Tanh normalization**: Smooth sigmoid function for score normalization
- **Directional adjustment**: Inverts scores for bearish factors (VIX, yields, etc.)

### 3. Weighted Aggregation

```
Weighted Score = Σ(Factor Score × Weight)
Bias Strength = (Total Weighted Score / Max Possible Score) × 100
Bias Confidence = (|Total Weighted Score| / Max Possible Score) × 100
```

### 4. Volatility & Regime

Market volatility is calculated from:
- VIX level (normalized to 0-100%)
- Treasury yield volatility (20-day standard deviation)
- Dollar index volatility
- Credit spread volatility

**Regime Classification:**
- Low Volatility: 0-33%
- Medium Volatility: 34-66%
- High Volatility: 67-100%

### 5. Factor Contributions

Each factor's contribution to total bias:
```
Contribution % = (|Factor Weighted Score| / Σ|All Weighted Scores|) × 100
```

## 🎨 Dashboard Integration

### Streamlit Dashboard

Create `streamlit_dashboard.py`:

```python
import streamlit as st
import plotly.graph_objects as go
from macro_bias_engine import MacroBiasEngine, OutputFormatter

st.set_page_config(page_title="Macro Bias Engine", layout="wide")
st.title("🌍 Macro Bias Engine")

# Initialize engine
engine = MacroBiasEngine()

# Run analysis
with st.spinner("Analyzing..."):
    results = engine.run_analysis()

# Display metrics
col1, col2, col3, col4 = st.columns(4)

summary = results['summary']
col1.metric("Overall Bias", summary['overall_bias'])
col2.metric("Strength", f"{summary['bias_strength_pct']:.1f}%")
col3.metric("Confidence", f"{summary['bias_confidence_pct']:.1f}%")
col4.metric("Volatility", f"{summary['volatility_pct']:.1f}%")

# Factor chart
import pandas as pd
df = pd.DataFrame(results['factor_scores'])
df = df.sort_values('Contribution_Pct', ascending=True)

fig = go.Figure(go.Bar(
    x=df['Contribution_Pct'],
    y=df['Factor'],
    orientation='h',
    marker=dict(color=df['Normalized_Score'], 
                colorscale='RdYlGn',
                cmin=-1, cmax=1)
))
st.plotly_chart(fig, use_container_width=True)

# Detailed table
st.dataframe(df)
```

Run with:
```bash
streamlit run streamlit_dashboard.py
```

### Flask API

Create `flask_api.py`:

```python
from flask import Flask, jsonify
from flask_cors import CORS
from macro_bias_engine import MacroBiasEngine, OutputFormatter

app = Flask(__name__)
CORS(app)

engine = MacroBiasEngine()

@app.route('/api/bias', methods=['GET'])
def get_bias():
    """API endpoint for macro bias analysis."""
    results = engine.run_analysis()
    dashboard_data = OutputFormatter.to_dashboard_dict(results)
    return jsonify(dashboard_data)

@app.route('/api/bias/detailed', methods=['GET'])
def get_bias_detailed():
    """API endpoint for detailed analysis."""
    results = engine.run_analysis()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Run with:
```bash
python flask_api.py
```

Access at: `http://localhost:5000/api/bias`

### React Integration

```javascript
import React, { useState, useEffect } from 'react';

function MacroBiasDashboard() {
  const [biasData, setBiasData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/api/bias')
      .then(response => response.json())
      .then(data => setBiasData(data))
      .catch(error => console.error('Error:', error));
  }, []);

  if (!biasData) return <div>Loading...</div>;

  return (
    <div className="macro-bias-dashboard">
      <h1>Macro Bias Engine</h1>
      <div className="metrics">
        <div className="metric">
          <h3>Overall Bias</h3>
          <p className={biasData.overall_bias.toLowerCase()}>
            {biasData.overall_bias}
          </p>
        </div>
        <div className="metric">
          <h3>Strength</h3>
          <p>{biasData.bias_strength.toFixed(1)}%</p>
        </div>
        <div className="metric">
          <h3>Confidence</h3>
          <p>{biasData.bias_confidence.toFixed(1)}%</p>
        </div>
        <div className="metric">
          <h3>Regime</h3>
          <p>{biasData.regime}</p>
        </div>
      </div>
      
      <div className="factors">
        <h2>Factor Breakdown</h2>
        {biasData.factors.map(factor => (
          <div key={factor.name} className="factor">
            <span>{factor.name}</span>
            <span className={factor.direction.toLowerCase()}>
              {factor.direction}
            </span>
            <span>{factor.contribution.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MacroBiasDashboard;
```

## 🔧 Customization

### Custom Weights

```python
custom_weights = {
    '10Y_Treasury_Yield': 2.5,      # Increase weight
    'DXY_Dollar_Index': 1.5,        # Decrease weight
    'M2_Money_Supply': 2.0,
    'Credit_Spreads_BAA_AAA': 2.0,  # Increase weight
    'VIX_Index': 1.0,               # Decrease weight
    'Economic_Surprises': 1.0
}

engine = MacroBiasEngine(weights=custom_weights)
results = engine.run_analysis()
```

### Custom Lookback Period

```python
from macro_bias_engine import MacroBiasEngine, MacroDataFetcher

# Fetch 180 days of data instead of default 90
fetcher = MacroDataFetcher(lookback_days=180)
factors_data = fetcher.get_all_factors()

# Use custom data with engine
engine = MacroBiasEngine()
# ... (manual scoring with custom data)
```

## 📊 Output Formats

### 1. Dashboard Dictionary

```python
dashboard_data = OutputFormatter.to_dashboard_dict(results)
# Returns:
{
    'timestamp': '2026-02-15T10:30:00',
    'overall_bias': 'Bullish',
    'bias_strength': 23.45,
    'bias_confidence': 67.89,
    'volatility': 45.23,
    'regime': 'Medium Volatility',
    'factors': [...]
}
```

### 2. JSON Export

```python
json_data = OutputFormatter.to_json(results)
# Save to file
with open('bias_results.json', 'w') as f:
    f.write(json_data)
```

### 3. DataFrame

```python
df = OutputFormatter.to_dataframe(results)
# Export to CSV
df.to_csv('factor_scores.csv', index=False)
```

## 🔍 Understanding the Scores

### Normalized Score (-1 to +1)

- **+1.0**: Extremely bullish signal
- **+0.5**: Moderately bullish
- **0.0**: Neutral
- **-0.5**: Moderately bearish
- **-1.0**: Extremely bearish signal

### Bias Strength (-100 to +100)

- **+50 to +100**: Strong bullish bias
- **+15 to +50**: Moderate bullish bias
- **-15 to +15**: Neutral
- **-50 to -15**: Moderate bearish bias
- **-100 to -50**: Strong bearish bias

### Confidence (0 to 100%)

- **80-100%**: Very high confidence in signal
- **60-80%**: High confidence
- **40-60%**: Moderate confidence
- **20-40%**: Low confidence
- **0-20%**: Very low confidence (likely noisy)

## 🛡️ Robustness Features

1. **Z-Score Normalization**: Standardizes factors relative to historical distribution
2. **Tanh Smoothing**: Prevents extreme outliers from dominating
3. **Weighted Aggregation**: Prioritizes more reliable factors
4. **Volatility Filtering**: Identifies noisy market conditions
5. **Factor Contribution**: Highlights dominant signals vs. weak noise

## 📝 Data Sources

- **FRED (Federal Reserve Economic Data)**: Free, no API key required
  - 10-Year Treasury Yield (DGS10)
  - M2 Money Supply (M2SL)
  - Corporate Bond Yields (DBAA, DAAA)

- **Yahoo Finance**: Free, via yfinance package
  - VIX Index (^VIX)
  - US Dollar Index (DX-Y.NYB)

## ⚠️ Important Notes

1. **Data Delays**: FRED data may have 1-2 day delays
2. **Market Hours**: Some Yahoo Finance data only updates during market hours
3. **API Limits**: No strict limits, but implement caching for production use
4. **Historical Analysis**: Engine requires 60+ days of data for accurate Z-scores

## 🚨 Troubleshooting

### "No module named 'yfinance'"

```bash
pip install yfinance pandas-datareader
```

### "KeyError: 'Close'"

Yahoo Finance ticker may be incorrect or data unavailable. Check ticker symbol.

### "Empty DataFrame"

FRED series may be delayed. Try increasing `lookback_days` or check FRED website.

### Network Errors

Ensure you have internet connection. FRED and Yahoo Finance require network access.

## 📚 Advanced Usage

### Batch Processing

```python
import schedule
import time

def run_analysis():
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    
    # Save to database or file
    with open(f'bias_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
        f.write(OutputFormatter.to_json(results))

# Run every 4 hours
schedule.every(4).hours.do(run_analysis)

while True:
    schedule.run_pending()
    time.sleep(1)
```

### Database Storage

```python
import sqlite3
import json

def save_to_db(results):
    conn = sqlite3.connect('macro_bias.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bias_history (
            timestamp TEXT PRIMARY KEY,
            overall_bias TEXT,
            bias_strength REAL,
            bias_confidence REAL,
            volatility REAL,
            regime TEXT,
            full_results TEXT
        )
    ''')
    
    cursor.execute('''
        INSERT INTO bias_history VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        results['timestamp'],
        results['summary']['overall_bias'],
        results['summary']['bias_strength_pct'],
        results['summary']['bias_confidence_pct'],
        results['summary']['volatility_pct'],
        results['summary']['regime'],
        json.dumps(results)
    ))
    
    conn.commit()
    conn.close()

# Usage
results = engine.run_analysis()
save_to_db(results)
```

## 🎓 Further Enhancements

Possible extensions to the engine:

1. **Machine Learning**: Train models on historical bias scores vs. market returns
2. **Backtesting**: Test bias signals against historical S&P 500 performance
3. **Alert System**: Send notifications when bias shifts significantly
4. **Multi-Asset**: Extend to currencies, commodities, crypto
5. **Sentiment Analysis**: Incorporate news sentiment as additional factor
6. **Options Positioning**: Add put/call ratios and skew metrics
7. **Technical Overlays**: Combine with trend indicators (RSI, MACD)

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests or open issues.

## 📧 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Disclaimer**: This tool is for informational purposes only. Not financial advice. Always do your own research and consult with financial professionals before making investment decisions.
