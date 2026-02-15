"""
Setup Historical Tracking
=========================

Run this once to initialize your historical bias database with the last 30 days of data.

Usage:
    python setup_history.py
"""

from historical_tracker import initial_setup

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       📊 MACRO BIAS ENGINE - HISTORICAL SETUP                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

This will set up historical tracking by:
  1. Creating the bias_history.db database
  2. Backfilling the last 30 days of macro bias data
  3. Giving you a full month of historical context

⏱️  This takes 3-5 minutes (fetching data for each past date)

💡 After setup, data will auto-update daily!
    """)
    
    try:
        initial_setup()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("  • Check your internet connection")
        print("  • Ensure yfinance and pandas-datareader are installed")
        print("  • Try running: pip install yfinance pandas-datareader")
