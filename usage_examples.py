"""
Macro Bias Engine - Usage Examples
===================================

This script demonstrates various use cases and integration patterns
for the Macro Bias Engine.

Run with: python usage_examples.py
"""

import json
from datetime import datetime
from macro_bias_engine import MacroBiasEngine, OutputFormatter

print("="*70)
print("MACRO BIAS ENGINE - COMPREHENSIVE USAGE EXAMPLES")
print("="*70)


# ============================================================================
# EXAMPLE 1: Basic Usage with Default Settings
# ============================================================================

def example_1_basic_usage():
    """
    Example 1: Basic usage with default weights.
    Perfect for getting started quickly.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Default Settings")
    print("="*70)
    
    # Initialize engine with default weights
    engine = MacroBiasEngine()
    
    # Run analysis
    results = engine.run_analysis()
    
    # Display formatted summary
    OutputFormatter.print_summary(results)
    
    # Access specific metrics
    summary = results['summary']
    print(f"\n📌 Quick Access:")
    print(f"   Bias: {summary['overall_bias']}")
    print(f"   Strength: {summary['bias_strength_pct']:.2f}%")
    print(f"   Confidence: {summary['bias_confidence_pct']:.2f}%")
    
    return results


# ============================================================================
# EXAMPLE 2: Custom Weights for Specific Trading Strategy
# ============================================================================

def example_2_custom_weights():
    """
    Example 2: Custom weights tailored to a specific trading strategy.
    
    Use case: You want to emphasize credit spreads and volatility
    for a corporate bond trading strategy.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Weights - Corporate Bond Focus")
    print("="*70)
    
    # Custom weights emphasizing credit and volatility
    custom_weights = {
        '10Y_Treasury_Yield': 2.5,      # Higher weight - important for bonds
        'DXY_Dollar_Index': 1.0,        # Lower weight - less relevant
        'M2_Money_Supply': 1.5,         # Moderate weight
        'Credit_Spreads_BAA_AAA': 3.0,  # Highest weight - most important
        'VIX_Index': 2.5,               # High weight - risk indicator
        'Economic_Surprises': 1.0       # Standard weight
    }
    
    print("\n📊 Custom Weights Applied:")
    for factor, weight in custom_weights.items():
        print(f"   {factor}: {weight}")
    
    # Initialize with custom weights
    engine = MacroBiasEngine(weights=custom_weights)
    results = engine.run_analysis()
    
    # Display results
    OutputFormatter.print_summary(results)
    
    return results


# ============================================================================
# EXAMPLE 3: Export to Different Formats
# ============================================================================

def example_3_export_formats(results):
    """
    Example 3: Export results in various formats for different use cases.
    
    Use case: You need to integrate with different systems or save results.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Export to Different Formats")
    print("="*70)
    
    # Format 1: JSON (for APIs, file storage)
    print("\n📄 JSON Format:")
    json_output = OutputFormatter.to_json(results)
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
    
    # Save to file
    with open('bias_results.json', 'w') as f:
        f.write(json_output)
    print("   ✅ Saved to bias_results.json")
    
    # Format 2: DataFrame (for data analysis, CSV export)
    print("\n📊 DataFrame Format:")
    df = OutputFormatter.to_dataframe(results)
    print(df.to_string())
    
    # Save to CSV
    df.to_csv('factor_scores.csv', index=False)
    print("   ✅ Saved to factor_scores.csv")
    
    # Format 3: Dashboard Dictionary (for web dashboards)
    print("\n🖥️  Dashboard Format:")
    dashboard_data = OutputFormatter.to_dashboard_dict(results)
    print(json.dumps(dashboard_data, indent=2)[:500] + "...")
    print("   ✅ Ready for web integration")
    
    return json_output, df, dashboard_data


# ============================================================================
# EXAMPLE 4: Interpreting Signals for Trading Decisions
# ============================================================================

def example_4_trading_signals(results):
    """
    Example 4: Convert bias analysis into actionable trading signals.
    
    Use case: You want clear buy/sell/neutral signals with confidence levels.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Actionable Trading Signals")
    print("="*70)
    
    summary = results['summary']
    bias_strength = summary['bias_strength_pct']
    confidence = summary['bias_confidence_pct']
    volatility = summary['volatility_pct']
    
    # Define signal rules
    print("\n🎯 Signal Generation Rules:")
    print("   Strong Signal: |Strength| > 30% AND Confidence > 60%")
    print("   Moderate Signal: |Strength| > 15% AND Confidence > 40%")
    print("   Weak Signal: |Strength| < 15% OR Confidence < 40%")
    
    # Generate signal
    signal = "NEUTRAL"
    signal_strength = "WEAK"
    
    if abs(bias_strength) > 30 and confidence > 60:
        signal = "STRONG BUY" if bias_strength > 0 else "STRONG SELL"
        signal_strength = "STRONG"
    elif abs(bias_strength) > 15 and confidence > 40:
        signal = "BUY" if bias_strength > 0 else "SELL"
        signal_strength = "MODERATE"
    else:
        signal = "HOLD"
        signal_strength = "WEAK"
    
    # Position sizing recommendation
    if volatility > 66:
        position_size = "Small (High Volatility)"
        risk_level = "HIGH"
    elif volatility > 33:
        position_size = "Normal (Medium Volatility)"
        risk_level = "MEDIUM"
    else:
        position_size = "Large (Low Volatility)"
        risk_level = "LOW"
    
    print("\n📈 TRADING RECOMMENDATION:")
    print(f"   Signal: {signal}")
    print(f"   Signal Strength: {signal_strength}")
    print(f"   Suggested Position Size: {position_size}")
    print(f"   Risk Level: {risk_level}")
    print(f"\n   Rationale:")
    print(f"   • Bias Strength: {bias_strength:+.2f}%")
    print(f"   • Confidence: {confidence:.2f}%")
    print(f"   • Volatility: {volatility:.2f}%")
    
    # Identify dominant factors
    import pandas as pd
    factors_df = pd.DataFrame(results['factor_scores'])
    factors_df = factors_df.sort_values('Contribution_Pct', ascending=False)
    
    print(f"\n   Key Drivers:")
    for idx, row in factors_df.head(3).iterrows():
        print(f"   • {row['Factor']}: {row['Direction']} ({row['Contribution_Pct']:.1f}% contribution)")
    
    return {
        'signal': signal,
        'strength': signal_strength,
        'position_size': position_size,
        'risk_level': risk_level
    }


# ============================================================================
# EXAMPLE 5: Risk Management Integration
# ============================================================================

def example_5_risk_management(results):
    """
    Example 5: Integrate bias analysis with risk management framework.
    
    Use case: Adjust portfolio risk based on macro conditions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Risk Management Framework")
    print("="*70)
    
    summary = results['summary']
    volatility = summary['volatility_pct']
    confidence = summary['bias_confidence_pct']
    regime = summary['regime']
    
    # Base portfolio allocation
    base_equity = 60  # 60% stocks
    base_bonds = 30   # 30% bonds
    base_cash = 10    # 10% cash
    
    print("\n📊 Base Portfolio Allocation:")
    print(f"   Equity: {base_equity}%")
    print(f"   Bonds: {base_bonds}%")
    print(f"   Cash: {base_cash}%")
    
    # Adjust based on macro bias and volatility
    if summary['overall_bias'] == 'Bullish' and confidence > 60 and volatility < 50:
        # Risk-on: Increase equity exposure
        adjusted_equity = min(base_equity + 10, 75)
        adjusted_bonds = base_bonds - 5
        adjusted_cash = base_cash - 5
        recommendation = "Risk-On (Favorable Conditions)"
    
    elif summary['overall_bias'] == 'Bearish' and confidence > 60:
        # Risk-off: Decrease equity exposure
        adjusted_equity = max(base_equity - 15, 40)
        adjusted_bonds = base_bonds + 5
        adjusted_cash = base_cash + 10
        recommendation = "Risk-Off (Defensive)"
    
    elif volatility > 66:
        # High volatility: Reduce risk regardless of bias
        adjusted_equity = max(base_equity - 10, 40)
        adjusted_bonds = base_bonds
        adjusted_cash = base_cash + 10
        recommendation = "Defensive (High Volatility)"
    
    else:
        # Neutral: Maintain base allocation
        adjusted_equity = base_equity
        adjusted_bonds = base_bonds
        adjusted_cash = base_cash
        recommendation = "Neutral (Maintain Current)"
    
    print("\n🎯 Recommended Adjustments:")
    print(f"   Strategy: {recommendation}")
    print(f"\n   Adjusted Allocation:")
    print(f"   Equity: {adjusted_equity}% ({adjusted_equity - base_equity:+}%)")
    print(f"   Bonds: {adjusted_bonds}% ({adjusted_bonds - base_bonds:+}%)")
    print(f"   Cash: {adjusted_cash}% ({adjusted_cash - base_cash:+}%)")
    
    print(f"\n   Rationale:")
    print(f"   • Macro Bias: {summary['overall_bias']}")
    print(f"   • Confidence: {confidence:.2f}%")
    print(f"   • Market Regime: {regime}")
    print(f"   • Volatility: {volatility:.2f}%")
    
    return {
        'equity': adjusted_equity,
        'bonds': adjusted_bonds,
        'cash': adjusted_cash,
        'recommendation': recommendation
    }


# ============================================================================
# EXAMPLE 6: Factor Contribution Analysis
# ============================================================================

def example_6_factor_analysis(results):
    """
    Example 6: Deep dive into which factors are driving the signal.
    
    Use case: Understand what's really moving the market.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Factor Contribution Analysis")
    print("="*70)
    
    import pandas as pd
    factors_df = pd.DataFrame(results['factor_scores'])
    factors_df = factors_df.sort_values('Contribution_Pct', ascending=False)
    
    print("\n🔍 Detailed Factor Analysis:")
    print("-" * 70)
    
    for idx, row in factors_df.iterrows():
        print(f"\n📌 {row['Factor']}")
        print(f"   Current Value: {row['Current_Value']:.2f}")
        print(f"   20-Day Change: {row['Percent_Change_20D']:+.2f}%")
        print(f"   Z-Score: {row['Z_Score']:+.2f} (standard deviations from mean)")
        print(f"   Normalized Score: {row['Normalized_Score']:+.2f}")
        print(f"   Direction: {row['Direction']}")
        print(f"   Weight: {row['Weight']}")
        print(f"   Weighted Score: {row['Weighted_Score']:+.2f}")
        print(f"   Contribution to Signal: {row['Contribution_Pct']:.2f}%")
        
        # Interpretation
        if abs(row['Z_Score']) > 2:
            print(f"   ⚠️  ALERT: Extreme value (|Z| > 2)")
        if row['Contribution_Pct'] > 30:
            print(f"   🎯 DOMINANT: This factor is driving the signal")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 Summary Statistics:")
    print(f"   Bullish Factors: {len(factors_df[factors_df['Direction'] == 'Bullish'])}")
    print(f"   Bearish Factors: {len(factors_df[factors_df['Direction'] == 'Bearish'])}")
    print(f"   Neutral Factors: {len(factors_df[factors_df['Direction'] == 'Neutral'])}")
    print(f"   Top Contributor: {factors_df.iloc[0]['Factor']} ({factors_df.iloc[0]['Contribution_Pct']:.1f}%)")


# ============================================================================
# EXAMPLE 7: Multi-Timeframe Analysis
# ============================================================================

def example_7_comparison():
    """
    Example 7: Compare default weights vs. custom weights.
    
    Use case: Sensitivity analysis for your strategy.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Strategy Comparison (Default vs Custom)")
    print("="*70)
    
    # Run with default weights
    print("\n🔵 Analysis with DEFAULT WEIGHTS:")
    engine_default = MacroBiasEngine()
    results_default = engine_default.run_analysis()
    
    # Run with aggressive weights (more risk-on/risk-off)
    print("\n🔴 Analysis with AGGRESSIVE WEIGHTS:")
    aggressive_weights = {
        '10Y_Treasury_Yield': 3.0,
        'DXY_Dollar_Index': 3.0,
        'M2_Money_Supply': 1.0,
        'Credit_Spreads_BAA_AAA': 2.0,
        'VIX_Index': 3.0,
        'Economic_Surprises': 1.0
    }
    engine_aggressive = MacroBiasEngine(weights=aggressive_weights)
    results_aggressive = engine_aggressive.run_analysis()
    
    # Compare results
    print("\n" + "="*70)
    print("📊 COMPARISON:")
    print("="*70)
    
    comparison = {
        'Metric': ['Overall Bias', 'Bias Strength', 'Confidence'],
        'Default Strategy': [
            results_default['summary']['overall_bias'],
            f"{results_default['summary']['bias_strength_pct']:.2f}%",
            f"{results_default['summary']['bias_confidence_pct']:.2f}%"
        ],
        'Aggressive Strategy': [
            results_aggressive['summary']['overall_bias'],
            f"{results_aggressive['summary']['bias_strength_pct']:.2f}%",
            f"{results_aggressive['summary']['bias_confidence_pct']:.2f}%"
        ]
    }
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    print("\n💡 Insight:")
    if results_default['summary']['overall_bias'] != results_aggressive['summary']['overall_bias']:
        print("   ⚠️  Different strategies produce different signals!")
        print("   Consider your risk tolerance and strategy objectives.")
    else:
        print("   ✅ Both strategies agree on the overall bias.")
        print("   Signal appears robust across different weighting schemes.")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all usage examples."""
    
    # Example 1: Basic usage
    results = example_1_basic_usage()
    
    # Example 2: Custom weights
    results_custom = example_2_custom_weights()
    
    # Example 3: Export formats
    json_output, df, dashboard_data = example_3_export_formats(results)
    
    # Example 4: Trading signals
    trading_signal = example_4_trading_signals(results)
    
    # Example 5: Risk management
    risk_allocation = example_5_risk_management(results)
    
    # Example 6: Factor analysis
    example_6_factor_analysis(results)
    
    # Example 7: Strategy comparison
    example_7_comparison()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n💡 Integration Tips:")
    print("   • Use these patterns in your own trading systems")
    print("   • Combine with your technical analysis")
    print("   • Adjust weights based on backtesting results")
    print("   • Monitor factor contributions for regime shifts")
    print("\n📚 Next Steps:")
    print("   • Try the Flask API (flask_api.py)")
    print("   • Build a dashboard (dashboard.html)")
    print("   • Backtest the signals against historical data")
    print("   • Integrate with your portfolio management system")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        run_all_examples()
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have installed the required packages:")
        print("pip install pandas numpy yfinance pandas-datareader")
