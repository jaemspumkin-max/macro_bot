"""
View Historical Bias Data
=========================

Standalone script to view, analyze, and export historical bias readings.

Usage:
    python view_history.py
"""

from historical_tracker import HistoricalBiasTracker, view_recent_history, compare_with_nasdaq
import pandas as pd
from datetime import datetime, timedelta
import sys


def print_menu():
    """Display the main menu."""
    print("\n" + "="*70)
    print("📊 MACRO BIAS ENGINE - HISTORICAL DATA VIEWER")
    print("="*70)
    print("\n1. View recent history (7 days)")
    print("2. View custom date range")
    print("3. View specific date")
    print("4. View statistics")
    print("5. Compare with Nasdaq (QQQ)")
    print("6. Export to CSV")
    print("7. View factor trends")
    print("8. Exit")
    print("\n" + "="*70)


def view_custom_range():
    """View history for a custom date range."""
    print("\n📅 Custom Date Range")
    print("-" * 70)
    
    days_str = input("Enter number of days (or press Enter for 30): ").strip()
    days = int(days_str) if days_str else 30
    
    tracker = HistoricalBiasTracker()
    history = tracker.get_history(days)
    
    if len(history) == 0:
        print("📭 No data available for this range")
        return
    
    print(f"\n📊 Bias History - Last {days} Days")
    print("=" * 100)
    
    for _, row in history.iterrows():
        bias_icon = "🟢" if row['overall_bias'] == 'Bullish' else "🔴" if row['overall_bias'] == 'Bearish' else "⚪"
        
        print(f"{row['date']} | {bias_icon} {row['overall_bias']:8s} | "
              f"Strength: {row['bias_strength']:+6.2f}% | "
              f"Confidence: {row['bias_confidence']:5.1f}% | "
              f"Vol: {row['volatility']:5.1f}% ({row['regime']})")
    
    print("=" * 100)


def view_specific_date():
    """View bias reading for a specific date."""
    print("\n📅 Specific Date Lookup")
    print("-" * 70)
    
    date_str = input("Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
    
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    tracker = HistoricalBiasTracker()
    data = tracker.get_specific_date(date_str)
    
    if data is None:
        print(f"\n📭 No data found for {date_str}")
        return
    
    print(f"\n📊 Bias Reading for {date_str}")
    print("=" * 70)
    print(f"Overall Bias:     {data['overall_bias']}")
    print(f"Bias Strength:    {data['bias_strength']:+.2f}%")
    print(f"Confidence:       {data['bias_confidence']:.2f}%")
    print(f"Volatility:       {data['volatility']:.2f}%")
    print(f"Regime:           {data['regime']}")
    print("\nFactor Breakdown:")
    print("-" * 70)
    
    for factor in data['factors']:
        direction_icon = "🟢" if factor['direction'] == 'Bullish' else "🔴" if factor['direction'] == 'Bearish' else "⚪"
        print(f"{direction_icon} {factor['factor_name']:30s} | "
              f"Score: {factor['normalized_score']:+.2f} | "
              f"Contribution: {factor['contribution_pct']:.1f}%")
    
    print("=" * 70)


def view_statistics():
    """View summary statistics."""
    print("\n📊 Statistical Summary")
    print("-" * 70)
    
    days_str = input("Enter number of days to analyze (or press Enter for 30): ").strip()
    days = int(days_str) if days_str else 30
    
    tracker = HistoricalBiasTracker()
    stats = tracker.get_statistics(days)
    
    if 'error' in stats:
        print(f"\n❌ {stats['error']}")
        return
    
    print(f"\n📈 Statistics for Last {stats['days_analyzed']} Days")
    print("=" * 70)
    
    print(f"\nDate Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    print("\n📊 Bias Distribution:")
    print(f"   Bullish Days:  {stats['bias_distribution']['bullish_days']} ({stats['bias_distribution']['bullish_days']/stats['days_analyzed']*100:.1f}%)")
    print(f"   Bearish Days:  {stats['bias_distribution']['bearish_days']} ({stats['bias_distribution']['bearish_days']/stats['days_analyzed']*100:.1f}%)")
    print(f"   Neutral Days:  {stats['bias_distribution']['neutral_days']} ({stats['bias_distribution']['neutral_days']/stats['days_analyzed']*100:.1f}%)")
    
    print("\n📊 Average Metrics:")
    print(f"   Bias Strength: {stats['average_metrics']['bias_strength']:+.2f}%")
    print(f"   Confidence:    {stats['average_metrics']['confidence']:.2f}%")
    print(f"   Volatility:    {stats['average_metrics']['volatility']:.2f}%")
    
    print("\n🎯 Strongest Signals:")
    print(f"   Most Bullish:  {stats['strongest_signals']['most_bullish']['date']} ({stats['strongest_signals']['most_bullish']['strength']:+.2f}%)")
    print(f"   Most Bearish:  {stats['strongest_signals']['most_bearish']['date']} ({stats['strongest_signals']['most_bearish']['strength']:+.2f}%)")
    
    print("\n📊 Regime Distribution:")
    for regime, count in stats['regime_distribution'].items():
        print(f"   {regime}: {count} days ({count/stats['days_analyzed']*100:.1f}%)")
    
    print("=" * 70)


def export_to_csv():
    """Export historical data to CSV."""
    print("\n💾 Export to CSV")
    print("-" * 70)
    
    days_str = input("Enter number of days to export (or press Enter for all): ").strip()
    days = int(days_str) if days_str else None
    
    filename = input("Enter filename (or press Enter for bias_history_export.csv): ").strip()
    if not filename:
        filename = 'bias_history_export.csv'
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    tracker = HistoricalBiasTracker()
    tracker.export_to_csv(filename, days)
    
    print(f"\n✅ Data exported to {filename}")


def view_factor_trends():
    """View trends for a specific factor."""
    print("\n📈 Factor Trends")
    print("-" * 70)
    
    print("\nAvailable factors:")
    print("1. 10Y_Treasury_Yield")
    print("2. DXY_Dollar_Index")
    print("3. M2_Money_Supply")
    print("4. Credit_Spreads_BAA_AAA")
    print("5. VIX_Index")
    
    choice = input("\nEnter factor number (or name): ").strip()
    
    factor_map = {
        '1': '10Y_Treasury_Yield',
        '2': 'DXY_Dollar_Index',
        '3': 'M2_Money_Supply',
        '4': 'Credit_Spreads_BAA_AAA',
        '5': 'VIX_Index'
    }
    
    factor_name = factor_map.get(choice, choice)
    
    days_str = input("Enter number of days (or press Enter for 30): ").strip()
    days = int(days_str) if days_str else 30
    
    tracker = HistoricalBiasTracker()
    history = tracker.get_factor_history(factor_name, days)
    
    if len(history) == 0:
        print(f"\n📭 No data available for {factor_name}")
        return
    
    print(f"\n📈 {factor_name} - Last {len(history)} Days")
    print("=" * 100)
    
    for _, row in history.iterrows():
        direction_icon = "🟢" if row['direction'] == 'Bullish' else "🔴" if row['direction'] == 'Bearish' else "⚪"
        
        print(f"{row['date']} | {direction_icon} {row['direction']:8s} | "
              f"Score: {row['normalized_score']:+.2f} | "
              f"Value: {row['current_value']:8.2f} | "
              f"Contribution: {row['contribution_pct']:5.1f}%")
    
    print("=" * 100)


def main():
    """Main menu loop."""
    tracker = HistoricalBiasTracker()
    
    while True:
        print_menu()
        choice = input("\nSelect an option (1-8): ").strip()
        
        try:
            if choice == '1':
                view_recent_history(days=7)
            
            elif choice == '2':
                view_custom_range()
            
            elif choice == '3':
                view_specific_date()
            
            elif choice == '4':
                view_statistics()
            
            elif choice == '5':
                print("\n📊 Comparing with Nasdaq (QQQ)...")
                compare_with_nasdaq(days=30)
            
            elif choice == '6':
                export_to_csv()
            
            elif choice == '7':
                view_factor_trends()
            
            elif choice == '8':
                print("\n👋 Goodbye!")
                sys.exit(0)
            
            else:
                print("\n❌ Invalid choice. Please enter 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
