"""
Macro Bias Engine for Trading
==============================

A quantitative engine that evaluates multiple macroeconomic and market factors
to determine directional bias, confidence, strength, volatility, and regime.

Author: Trading Analytics
Version: 1.0.1 (Fixed)
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
# HELPER FUNCTION FOR SAFE SERIES CHECKING
# ============================================================================

def is_valid_series(series):
    """
    Safely check if a series is valid and has data.
    
    Args:
        series: pandas Series or None
        
    Returns:
        bool: True if series is valid and has data
    """
    if series is None:
        return False
    if not isinstance(series, pd.Series):
        return False
    if len(series) == 0:
        return False
    try:
        if series.empty:
            return False
    except:
        pass
    return True


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
            pd.Series: Time series data or None
        """
        if not DATAREADER_AVAILABLE:
            return None
        
        try:
            data = pdr.DataReader(series_id, 'fred', self.start_date, self.end_date)
            if isinstance(data, pd.DataFrame):
                # Get first column as Series
                series = data.iloc[:, 0]
            else:
                series = data
            
            # Drop NaN values
            series = series.dropna()
            
            if len(series) > 0:
                return series
            else:
                return None
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None
    
    def fetch_yahoo_data(self, ticker):
        """
        Fetch data from Yahoo Finance.
        
        Args:
            ticker (str): Yahoo Finance ticker symbol
            
        Returns:
            pd.Series: Closing price time series or None
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, 
                             progress=False)
            if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                series = data['Close']
                # Drop NaN values
                series = series.dropna()
                if len(series) > 0:
                    return series
            return None
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
        if is_valid_series(baa) and is_valid_series(aaa):
            # Align the series
            combined = pd.concat([baa, aaa], axis=1).dropna()
            if len(combined) > 0:
                factors['Credit_Spreads_BAA_AAA'] = combined.iloc[:, 0] - combined.iloc[:, 1]
            else:
                factors['Credit_Spreads_BAA_AAA'] = None
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
        if not is_valid_series(series):
            return 0.0
        
        if len(series) < periods:
            return 0.0
        
        try:
            current = float(series.iloc[-1])
            past = float(series.iloc[-periods])
            
            if pd.isna(current) or pd.isna(past) or past == 0:
                return 0.0
            
            return (current - past) / past * 100
        except:
            return 0.0
    
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
        if not is_valid_series(series):
            return 0.0
        
        if len(series) < window:
            return 0.0
        
        try:
            recent_data = series.iloc[-window:]
            mean = float(recent_data.mean())
            std = float(recent_data.std())
            
            if pd.isna(std) or std == 0:
                return 0.0
            
            current = float(series.iloc[-1])
            if pd.isna(current):
                return 0.0
            
            z_score = (current - mean) / std
            
            return float(z_score)
        except:
            return 0.0
    
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
        if not is_valid_series(series):
            return 0.0, 0.0, 0.0, None
        
        try:
            # Calculate metrics
            pct_change = self.calculate_change(series)
            z_score = self.calculate_z_score(series)
            current_value = float(series.iloc[-1])
            
            # Normalize Z-score to -1 to 1 range using tanh function
            # tanh provides a smooth sigmoid-like normalization
            normalized_score = np.tanh(z_score / 2)  # Divide by 2 to make it less sensitive
            
            # Invert if needed (for bearish factors like VIX, yields, DXY, credit spreads)
            if invert:
                normalized_score = -normalized_score
            
            # Clamp to -1 to 1 range
            normalized_score = np.clip(normalized_score, -1, 1)
            
            return float(normalized_score), float(pct_change), float(z_score), float(current_value)
        except Exception as e:
            print(f"Error scoring {factor_name}: {e}")
            return 0.0, 0.0, 0.0, None
    
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
            if not is_valid_series(series):
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
        
        if len(results) == 0:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=[
                'Factor', 'Current_Value', 'Percent_Change_20D', 'Z_Score',
                'Normalized_Score', 'Weight', 'Weighted_Score', 'Direction'
            ])
        
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
        if len(scores_df) == 0:
            return {
                'overall_bias': 'Neutral',
                'bias_strength': 0.0,
                'bias_confidence': 0.0,
                'total_weighted_score': 0.0,
                'max_possible_score': 0.0
            }
        
        # Total weighted score
        total_weighted_score = float(scores_df['Weighted_Score'].sum())
        
        # Maximum possible weighted score (all factors at +1 or -1)
        max_possible_score = float(scores_df['Weight'].sum())
        
        if max_possible_score == 0:
            return {
                'overall_bias': 'Neutral',
                'bias_strength': 0.0,
                'bias_confidence': 0.0,
                'total_weighted_score': 0.0,
                'max_possible_score': 0.0
            }
        
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
            'bias_strength': round(float(bias_strength), 2),
            'bias_confidence': round(float(bias_confidence), 2),
            'total_weighted_score': round(float(total_weighted_score), 4),
            'max_possible_score': round(float(max_possible_score), 4)
        }
    
    def calculate_factor_contributions(self, scores_df):
        """
        Calculate each factor's contribution to the total bias.
        
        Args:
            scores_df (pd.DataFrame): DataFrame with factor scores
            
        Returns:
            pd.DataFrame: Factor contributions
        """
        if len(scores_df) == 0:
            return scores_df
        
        total_abs_weighted = float(scores_df['Weighted_Score'].abs().sum())
        
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
        if 'VIX_Index' in factors_data and is_valid_series(factors_data['VIX_Index']):
            try:
                vix_current = float(factors_data['VIX_Index'].iloc[-1])
                vix_normalized = min(vix_current / 50 * 100, 100)  # VIX of 50+ = 100%
                volatility_components.append(vix_normalized)
            except:
                pass
        
        # 10Y Yield volatility (standard deviation)
        if '10Y_Treasury_Yield' in factors_data and is_valid_series(factors_data['10Y_Treasury_Yield']):
            try:
                yield_vol = float(factors_data['10Y_Treasury_Yield'].iloc[-20:].std())
                yield_vol_normalized = min(yield_vol / 0.5 * 100, 100)  # 0.5% std = 100%
                volatility_components.append(yield_vol_normalized)
            except:
                pass
        
        # DXY volatility
        if 'DXY_Dollar_Index' in factors_data and is_valid_series(factors_data['DXY_Dollar_Index']):
            try:
                dxy_vol = float(factors_data['DXY_Dollar_Index'].pct_change().iloc[-20:].std() * 100)
                dxy_vol_normalized = min(dxy_vol / 2 * 100, 100)  # 2% std = 100%
                volatility_components.append(dxy_vol_normalized)
            except:
                pass
        
        # Credit spread changes
        if 'Credit_Spreads_BAA_AAA' in factors_data and is_valid_series(factors_data['Credit_Spreads_BAA_AAA']):
            try:
                spread_vol = float(factors_data['Credit_Spreads_BAA_AAA'].iloc[-20:].std())
                spread_vol_normalized = min(spread_vol / 0.3 * 100, 100)  # 0.3% std = 100%
                volatility_components.append(spread_vol_normalized)
            except:
                pass
        
        # Average volatility
        if volatility_components:
            volatility_pct = float(np.mean(volatility_components))
        else:
            volatility_pct = 50.0  # Default to medium
        
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
        if len(scores_df) > 0:
            scores_df = scores_df.sort_values('Contribution_Pct', ascending=False)
            
            for _, row in scores_df.iterrows():
                direction_icon = "🟢" if row['Direction'] == 'Bullish' else "🔴" if row['Direction'] == 'Bearish' else "⚪"
                print(f"{direction_icon} {row['Factor']:30s} | "
                      f"Score: {row['Normalized_Score']:+.2f} | "
                      f"Weight: {row['Weight']:.1f} | "
                      f"Contribution: {row['Contribution_Pct']:.1f}%")
        else:
            print("⚠️  No factor data available")
        
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
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if required packages are available
    if not YFINANCE_AVAILABLE or not DATAREADER_AVAILABLE:
        print("\n⚠️  INSTALLATION REQUIRED ⚠️")
        print("Please install required packages:")
        print("pip install yfinance pandas-datareader")
    else:
        # Run example
        engine = MacroBiasEngine()
        results = engine.run_analysis()
        OutputFormatter.print_summary(results)
