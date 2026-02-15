"""
Macro Bias Engine for Trading
==============================

A quantitative engine that evaluates multiple macroeconomic and market factors
to determine directional bias, confidence, strength, volatility, and regime.

Author: Trading Analytics
Version: 1.0
Date: 2026-02-15
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports - will handle gracefully if not installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")

try:
    from pandas_datareader import data as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("Warning: pandas-datareader not installed. Install with: pip install pandas-datareader")


# ============================================================================
# CONFIGURATION & WEIGHTS
# ============================================================================

DEFAULT_FACTOR_WEIGHTS = {
    '10Y_Treasury_Yield': 2.0,      # Rising yields = bearish (tightening)
    'DXY_Dollar_Index': 2.0,         # Rising dollar = bearish (tight conditions)
    'M2_Money_Supply': 2.0,          # Rising M2 = bullish (liquidity)
    'Credit_Spreads_BAA_AAA': 1.5,   # Widening spreads = bearish (stress)
    'VIX_Index': 1.5,                # Rising VIX = bearish (fear)
    'Economic_Surprises': 1.0        # Positive surprises = bullish
}

# FRED API Configuration (free, no key needed for basic use)
FRED_SERIES = {
    '10Y_Treasury_Yield': 'DGS10',        # 10-Year Treasury Constant Maturity Rate
    'M2_Money_Supply': 'M2SL',            # M2 Money Stock
    'BAA_Yield': 'DBAA',                  # Moody's Seasoned Baa Corporate Bond Yield
    'AAA_Yield': 'DAAA',                  # Moody's Seasoned Aaa Corporate Bond Yield
}

# Volatility regime thresholds
VOLATILITY_THRESHOLDS = {
    'Low': (0, 33),
    'Medium': (34, 66),
    'High': (67, 100)
}


# ============================================================================
# DATA FETCHING MODULE
# ============================================================================

class MacroDataFetcher:
    """Fetch macroeconomic and market data from various free sources."""
    
    def __init__(self, lookback_days=90):
        """
        Initialize the data fetcher.
        
        Args:
            lookback_days (int): Number of days of historical data to fetch
        """
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
    
    def fetch_fred_data(self, series_id):
        """
        Fetch data from FRED (Federal Reserve Economic Data).
        
        Args:
            series_id (str): FRED series identifier
            
        Returns:
            pd.Series: Time series data
        """
        if not DATAREADER_AVAILABLE:
            return None
        
        try:
            data = pdr.DataReader(series_id, 'fred', self.start_date, self.end_date)
            return data.iloc[:, 0]  # Return the series
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None
    
    def fetch_yahoo_data(self, ticker):
        """
        Fetch data from Yahoo Finance.
        
        Args:
            ticker (str): Yahoo Finance ticker symbol
            
        Returns:
            pd.Series: Closing price time series
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, 
                             progress=False)
            return data['Close']
        except Exception as e:
            print(f"Error fetching Yahoo data for {ticker}: {e}")
            return None
    
    def get_all_factors(self):
        """
        Fetch all required factors for the macro bias engine.
        
        Returns:
            dict: Dictionary of factor time series
        """
        factors = {}
        
        # 10-Year Treasury Yield
        print("Fetching 10-Year Treasury Yield...")
        factors['10Y_Treasury_Yield'] = self.fetch_fred_data(FRED_SERIES['10Y_Treasury_Yield'])
        
        # US Dollar Index (DXY)
        print("Fetching US Dollar Index (DXY)...")
        factors['DXY_Dollar_Index'] = self.fetch_yahoo_data('DX-Y.NYB')
        
        # M2 Money Supply
        print("Fetching M2 Money Supply...")
        factors['M2_Money_Supply'] = self.fetch_fred_data(FRED_SERIES['M2_Money_Supply'])
        
        # Credit Spreads (BAA - AAA)
        print("Fetching Credit Spreads...")
        baa = self.fetch_fred_data(FRED_SERIES['BAA_Yield'])
        aaa = self.fetch_fred_data(FRED_SERIES['AAA_Yield'])
        if baa is not None and aaa is not None:
            factors['Credit_Spreads_BAA_AAA'] = baa - aaa
        else:
            factors['Credit_Spreads_BAA_AAA'] = None
        
        # VIX Index
        print("Fetching VIX Index...")
        factors['VIX_Index'] = self.fetch_yahoo_data('^VIX')
        
        # Economic Surprises (placeholder - would need Citi Economic Surprise Index)
        # For now, we'll use a synthetic indicator or leave as optional
        factors['Economic_Surprises'] = None
        
        return factors


# ============================================================================
# FACTOR SCORING MODULE
# ============================================================================

class FactorScorer:
    """Normalize and score individual factors on a -1 to 1 scale."""
    
    @staticmethod
    def calculate_change(series, periods=20):
        """
        Calculate percentage change over specified periods.
        
        Args:
            series (pd.Series): Time series data
            periods (int): Number of periods for change calculation
            
        Returns:
            float: Percentage change
        """
        if series is None or len(series) < periods:
            return 0.0
        
        current = series.iloc[-1]
        past = series.iloc[-periods]
        
        # Handle scalar comparison properly
        if pd.isna(past) or past == 0 or pd.isna(current):
            return 0.0
        
        return (current - past) / past * 100
    
    @staticmethod
    def calculate_z_score(series, window=60):
        """
        Calculate Z-score (standardized score) for recent value.
        
        Args:
            series (pd.Series): Time series data
            window (int): Lookback window for mean/std calculation
            
        Returns:
            float: Z-score
        """
        if series is None or len(series) < window:
            return 0.0
        
        recent_data = series.iloc[-window:]
        mean = recent_data.mean()
        std = recent_data.std()
        
        if std == 0:
            return 0.0
        
        current = series.iloc[-1]
        z_score = (current - mean) / std
        
        return z_score
    
    def score_factor(self, series, factor_name, invert=False):
        """
        Score a factor on a -1 to 1 scale.
        
        Args:
            series (pd.Series): Time series data
            factor_name (str): Name of the factor
            invert (bool): If True, invert the score (e.g., for bearish factors)
            
        Returns:
            tuple: (score, raw_change, z_score, current_value)
        """
        if series is None or (hasattr(series, 'empty') and series.empty) or len(series) == 0:
            return 0.0, 0.0, 0.0, None
        
        # Calculate metrics
        pct_change = self.calculate_change(series)
        z_score = self.calculate_z_score(series)
        current_value = series.iloc[-1]
        
        # Normalize Z-score to -1 to 1 range using tanh function
        # tanh provides a smooth sigmoid-like normalization
        normalized_score = np.tanh(z_score / 2)  # Divide by 2 to make it less sensitive
        
        # Invert if needed (for bearish factors like VIX, yields, DXY, credit spreads)
        if invert:
            normalized_score = -normalized_score
        
        # Clamp to -1 to 1 range
        normalized_score = np.clip(normalized_score, -1, 1)
        
        return normalized_score, pct_change, z_score, current_value
    
    def score_all_factors(self, factors_data, weights):
        """
        Score all factors and prepare for aggregation.
        
        Args:
            factors_data (dict): Dictionary of factor time series
            weights (dict): Dictionary of factor weights
            
        Returns:
            pd.DataFrame: DataFrame with scores and metrics for each factor
        """
        results = []
        
        # Factor scoring rules:
        # Bullish when rising: M2_Money_Supply
        # Bearish when rising: 10Y_Treasury_Yield, DXY_Dollar_Index, Credit_Spreads, VIX
        
        bearish_factors = [
            '10Y_Treasury_Yield',
            'DXY_Dollar_Index',
            'Credit_Spreads_BAA_AAA',
            'VIX_Index'
        ]
        
        for factor_name, series in factors_data.items():
            if series is None or (hasattr(series, 'empty') and series.empty):
                continue
            
            # Determine if factor should be inverted (bearish when rising)
            invert = factor_name in bearish_factors
            
            # Score the factor
            score, pct_change, z_score, current_value = self.score_factor(
                series, factor_name, invert=invert
            )
            
            weight = weights.get(factor_name, 1.0)
            weighted_score = score * weight
            
            results.append({
                'Factor': factor_name,
                'Current_Value': current_value,
                'Percent_Change_20D': pct_change,
                'Z_Score': z_score,
                'Normalized_Score': score,
                'Weight': weight,
                'Weighted_Score': weighted_score,
                'Direction': 'Bullish' if score > 0 else 'Bearish' if score < 0 else 'Neutral'
            })
        
        return pd.DataFrame(results)


# ============================================================================
# BIAS CALCULATION ENGINE
# ============================================================================

class MacroBiasEngine:
    """Main engine for calculating macro bias, confidence, and regime."""
    
    def __init__(self, weights=None):
        """
        Initialize the bias engine.
        
        Args:
            weights (dict): Custom factor weights (optional)
        """
        self.weights = weights if weights is not None else DEFAULT_FACTOR_WEIGHTS
        self.fetcher = MacroDataFetcher()
        self.scorer = FactorScorer()
    
    def calculate_bias_metrics(self, scores_df):
        """
        Calculate overall bias metrics from factor scores.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with factor scores
            
        Returns:
            dict: Bias metrics including bias, confidence, strength
        """
        # Total weighted score
        total_weighted_score = scores_df['Weighted_Score'].sum()
        
        # Maximum possible weighted score (all factors at +1 or -1)
        max_possible_score = scores_df['Weight'].sum()
        
        # Bias strength: -100 to +100
        # Positive = Bullish, Negative = Bearish
        bias_strength = (total_weighted_score / max_possible_score) * 100
        
        # Bias confidence: 0 to 100
        # How strong the signal is regardless of direction
        bias_confidence = (abs(total_weighted_score) / max_possible_score) * 100
        
        # Overall bias classification
        if bias_strength > 15:
            overall_bias = 'Bullish'
        elif bias_strength < -15:
            overall_bias = 'Bearish'
        else:
            overall_bias = 'Neutral'
        
        return {
            'overall_bias': overall_bias,
            'bias_strength': round(bias_strength, 2),
            'bias_confidence': round(bias_confidence, 2),
            'total_weighted_score': round(total_weighted_score, 4),
            'max_possible_score': round(max_possible_score, 4)
        }
    
    def calculate_factor_contributions(self, scores_df):
        """
        Calculate each factor's contribution to the total bias.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with factor scores
            
        Returns:
            pd.DataFrame: Factor contributions
        """
        total_abs_weighted = scores_df['Weighted_Score'].abs().sum()
        
        if total_abs_weighted == 0:
            scores_df['Contribution_Pct'] = 0
        else:
            scores_df['Contribution_Pct'] = (
                scores_df['Weighted_Score'].abs() / total_abs_weighted * 100
            ).round(2)
        
        return scores_df
    
    def calculate_volatility_regime(self, factors_data, scores_df):
        """
        Calculate market volatility and regime classification.
        
        Args:
            factors_data (dict): Raw factor time series
            scores_df (pd.DataFrame): Scored factors
            
        Returns:
            dict: Volatility metrics and regime
        """
        # Components for volatility calculation
        volatility_components = []
        
        # VIX level (normalized)
        if 'VIX_Index' in factors_data and factors_data['VIX_Index'] is not None:
            vix_current = factors_data['VIX_Index'].iloc[-1]
            vix_normalized = min(vix_current / 50 * 100, 100)  # VIX of 50+ = 100%
            volatility_components.append(vix_normalized)
        
        # 10Y Yield volatility (standard deviation)
        if '10Y_Treasury_Yield' in factors_data and factors_data['10Y_Treasury_Yield'] is not None:
            yield_vol = factors_data['10Y_Treasury_Yield'].iloc[-20:].std()
            yield_vol_normalized = min(yield_vol / 0.5 * 100, 100)  # 0.5% std = 100%
            volatility_components.append(yield_vol_normalized)
        
        # DXY volatility
        if 'DXY_Dollar_Index' in factors_data and factors_data['DXY_Dollar_Index'] is not None:
            dxy_vol = factors_data['DXY_Dollar_Index'].pct_change().iloc[-20:].std() * 100
            dxy_vol_normalized = min(dxy_vol / 2 * 100, 100)  # 2% std = 100%
            volatility_components.append(dxy_vol_normalized)
        
        # Credit spread changes
        if 'Credit_Spreads_BAA_AAA' in factors_data and factors_data['Credit_Spreads_BAA_AAA'] is not None:
            spread_vol = factors_data['Credit_Spreads_BAA_AAA'].iloc[-20:].std()
            spread_vol_normalized = min(spread_vol / 0.3 * 100, 100)  # 0.3% std = 100%
            volatility_components.append(spread_vol_normalized)
        
        # Average volatility
        if volatility_components:
            volatility_pct = np.mean(volatility_components)
        else:
            volatility_pct = 50  # Default to medium
        
        # Classify regime
        if volatility_pct <= VOLATILITY_THRESHOLDS['Low'][1]:
            regime = 'Low Volatility'
        elif volatility_pct <= VOLATILITY_THRESHOLDS['Medium'][1]:
            regime = 'Medium Volatility'
        else:
            regime = 'High Volatility'
        
        return {
            'volatility_pct': round(volatility_pct, 2),
            'regime': regime,
            'volatility_components': {
                'vix_component': round(volatility_components[0], 2) if len(volatility_components) > 0 else None,
                'yield_component': round(volatility_components[1], 2) if len(volatility_components) > 1 else None,
                'dxy_component': round(volatility_components[2], 2) if len(volatility_components) > 2 else None,
                'spread_component': round(volatility_components[3], 2) if len(volatility_components) > 3 else None,
            }
        }
    
    def run_analysis(self):
        """
        Run complete macro bias analysis.
        
        Returns:
            dict: Complete analysis results
        """
        print("\n" + "="*70)
        print("MACRO BIAS ENGINE - ANALYSIS STARTING")
        print("="*70)
        
        # Step 1: Fetch data
        print("\n[1/4] Fetching macroeconomic data...")
        factors_data = self.fetcher.get_all_factors()
        
        # Step 2: Score factors
        print("\n[2/4] Scoring factors...")
        scores_df = self.scorer.score_all_factors(factors_data, self.weights)
        
        # Step 3: Calculate bias metrics
        print("\n[3/4] Calculating bias metrics...")
        bias_metrics = self.calculate_bias_metrics(scores_df)
        
        # Step 4: Calculate volatility and regime
        print("\n[4/4] Calculating volatility and regime...")
        volatility_metrics = self.calculate_volatility_regime(factors_data, scores_df)
        
        # Calculate factor contributions
        scores_df = self.calculate_factor_contributions(scores_df)
        
        # Compile complete results
        results = {
            'timestamp': datetime.now().isoformat(),
            'bias_metrics': bias_metrics,
            'volatility_metrics': volatility_metrics,
            'factor_scores': scores_df.to_dict('records'),
            'summary': {
                'overall_bias': bias_metrics['overall_bias'],
                'bias_strength_pct': bias_metrics['bias_strength'],
                'bias_confidence_pct': bias_metrics['bias_confidence'],
                'volatility_pct': volatility_metrics['volatility_pct'],
                'regime': volatility_metrics['regime']
            }
        }
        
        return results


# ============================================================================
# OUTPUT FORMATTING & DISPLAY
# ============================================================================

class OutputFormatter:
    """Format and display analysis results."""
    
    @staticmethod
    def print_summary(results):
        """Print formatted summary of results."""
        print("\n" + "="*70)
        print("MACRO BIAS ENGINE - RESULTS SUMMARY")
        print("="*70)
        
        summary = results['summary']
        bias = results['bias_metrics']
        volatility = results['volatility_metrics']
        
        print(f"\n📊 OVERALL BIAS: {summary['overall_bias']}")
        print(f"   Strength: {summary['bias_strength_pct']:.2f}% " +
              f"({'Bullish' if summary['bias_strength_pct'] > 0 else 'Bearish'})")
        print(f"   Confidence: {summary['bias_confidence_pct']:.2f}%")
        
        print(f"\n🌊 MARKET REGIME: {summary['regime']}")
        print(f"   Volatility: {summary['volatility_pct']:.2f}%")
        
        print("\n📈 FACTOR BREAKDOWN:")
        print("-" * 70)
        
        scores_df = pd.DataFrame(results['factor_scores'])
        scores_df = scores_df.sort_values('Contribution_Pct', ascending=False)
        
        for _, row in scores_df.iterrows():
            direction_icon = "🟢" if row['Direction'] == 'Bullish' else "🔴" if row['Direction'] == 'Bearish' else "⚪"
            print(f"{direction_icon} {row['Factor']:30s} | "
                  f"Score: {row['Normalized_Score']:+.2f} | "
                  f"Weight: {row['Weight']:.1f} | "
                  f"Contribution: {row['Contribution_Pct']:.1f}%")
        
        print("\n" + "="*70)
    
    @staticmethod
    def to_json(results):
        """Convert results to JSON string."""
        import json
        return json.dumps(results, indent=2, default=str)
    
    @staticmethod
    def to_dataframe(results):
        """Convert factor scores to DataFrame."""
        return pd.DataFrame(results['factor_scores'])
    
    @staticmethod
    def to_dashboard_dict(results):
        """
        Format results for dashboard integration.
        
        Returns a clean dictionary optimized for web display.
        """
        return {
            'timestamp': results['timestamp'],
            'overall_bias': results['summary']['overall_bias'],
            'bias_strength': results['summary']['bias_strength_pct'],
            'bias_confidence': results['summary']['bias_confidence_pct'],
            'volatility': results['summary']['volatility_pct'],
            'regime': results['summary']['regime'],
            'factors': [
                {
                    'name': f['Factor'],
                    'score': f['Normalized_Score'],
                    'direction': f['Direction'],
                    'contribution': f['Contribution_Pct'],
                    'current_value': f['Current_Value']
                }
                for f in sorted(results['factor_scores'], 
                              key=lambda x: x['Contribution_Pct'], 
                              reverse=True)
            ]
        }


# ============================================================================
# EXAMPLE USAGE & DEMO
# ============================================================================

def example_usage():
    """
    Example demonstrating how to use the Macro Bias Engine.
    """
    print("\n" + "="*70)
    print("MACRO BIAS ENGINE - EXAMPLE USAGE")
    print("="*70)
    
    # Example 1: Use default weights
    print("\n### EXAMPLE 1: Using Default Weights ###")
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    
    # Print formatted summary
    OutputFormatter.print_summary(results)
    
    # Example 2: Custom weights
    print("\n\n### EXAMPLE 2: Using Custom Weights ###")
    custom_weights = {
        '10Y_Treasury_Yield': 2.5,
        'DXY_Dollar_Index': 1.5,
        'M2_Money_Supply': 2.0,
        'Credit_Spreads_BAA_AAA': 2.0,
        'VIX_Index': 1.0,
        'Economic_Surprises': 1.0
    }
    
    engine_custom = MacroBiasEngine(weights=custom_weights)
    results_custom = engine_custom.run_analysis()
    OutputFormatter.print_summary(results_custom)
    
    # Example 3: Export for dashboard
    print("\n\n### EXAMPLE 3: Dashboard Integration Format ###")
    dashboard_data = OutputFormatter.to_dashboard_dict(results)
    print("\nDashboard-ready dictionary:")
    import json
    print(json.dumps(dashboard_data, indent=2, default=str))
    
    # Example 4: Export to DataFrame
    print("\n\n### EXAMPLE 4: DataFrame Export ###")
    df = OutputFormatter.to_dataframe(results)
    print("\nFactor Scores DataFrame:")
    print(df.to_string())
    
    return results


# ============================================================================
# STREAMLIT DASHBOARD EXAMPLE
# ============================================================================

def create_streamlit_dashboard():
    """
    Example Streamlit dashboard code.
    Save this section to a separate file: streamlit_dashboard.py
    Run with: streamlit run streamlit_dashboard.py
    """
    
    dashboard_code = '''
import streamlit as st
import plotly.graph_objects as go
from macro_bias_engine import MacroBiasEngine, OutputFormatter

st.set_page_config(page_title="Macro Bias Engine", layout="wide")

st.title("🌍 Macro Bias Engine - Real-Time Market Analysis")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
custom_weights = st.sidebar.checkbox("Use Custom Weights")

if custom_weights:
    weights = {
        '10Y_Treasury_Yield': st.sidebar.slider("10Y Yield", 0.0, 3.0, 2.0),
        'DXY_Dollar_Index': st.sidebar.slider("DXY", 0.0, 3.0, 2.0),
        'M2_Money_Supply': st.sidebar.slider("M2", 0.0, 3.0, 2.0),
        'Credit_Spreads_BAA_AAA': st.sidebar.slider("Credit Spreads", 0.0, 3.0, 1.5),
        'VIX_Index': st.sidebar.slider("VIX", 0.0, 3.0, 1.5),
    }
    engine = MacroBiasEngine(weights=weights)
else:
    engine = MacroBiasEngine()

# Run analysis
if st.sidebar.button("🔄 Refresh Analysis"):
    with st.spinner("Analyzing macro factors..."):
        results = engine.run_analysis()
        st.session_state['results'] = results

if 'results' not in st.session_state:
    with st.spinner("Running initial analysis..."):
        results = engine.run_analysis()
        st.session_state['results'] = results

results = st.session_state['results']
summary = results['summary']

# Display main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    bias_color = "🟢" if summary['overall_bias'] == 'Bullish' else "🔴" if summary['overall_bias'] == 'Bearish' else "⚪"
    st.metric("Overall Bias", f"{bias_color} {summary['overall_bias']}")

with col2:
    st.metric("Bias Strength", f"{summary['bias_strength_pct']:.1f}%")

with col3:
    st.metric("Confidence", f"{summary['bias_confidence_pct']:.1f}%")

with col4:
    st.metric("Volatility", f"{summary['volatility_pct']:.1f}%", 
              delta=summary['regime'])

# Factor contributions chart
st.subheader("📊 Factor Contributions")
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
fig.update_layout(height=400, xaxis_title="Contribution %", yaxis_title="Factor")
st.plotly_chart(fig, use_container_width=True)

# Detailed factor table
st.subheader("📈 Detailed Factor Analysis")
st.dataframe(df[['Factor', 'Direction', 'Normalized_Score', 'Weight', 
                 'Contribution_Pct', 'Current_Value']], 
             use_container_width=True)
'''
    
    return dashboard_code


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if required packages are available
    if not YFINANCE_AVAILABLE or not DATAREADER_AVAILABLE:
        print("\n⚠️  INSTALLATION REQUIRED ⚠️")
        print("Please install required packages:")
        print("pip install yfinance pandas-datareader")
        print("\nOr install all at once:")
        print("pip install yfinance pandas-datareader plotly streamlit")
    else:
        # Run the example
        results = example_usage()
        
        print("\n\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\n💡 INTEGRATION TIPS:")
        print("   • Use OutputFormatter.to_dashboard_dict() for web dashboards")
        print("   • Use OutputFormatter.to_json() for API responses")
        print("   • Use OutputFormatter.to_dataframe() for data analysis")
        print("\n📊 For Streamlit dashboard:")
        print("   • Save the dashboard code to streamlit_dashboard.py")
        print("   • Run: streamlit run streamlit_dashboard.py")
        print("\n" + "="*70)
