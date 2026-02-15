"""
Historical Bias Tracker
=======================

Stores and retrieves historical macro bias readings for analysis and backtesting.

Author: Trading Analytics
Version: 1.0
Date: 2026-02-15
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import sqlite3


class HistoricalBiasTracker:
    """Track and store historical macro bias readings."""
    
    def __init__(self, db_path='bias_history.db'):
        """
        Initialize the historical tracker.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main bias history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bias_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                overall_bias TEXT NOT NULL,
                bias_strength REAL NOT NULL,
                bias_confidence REAL NOT NULL,
                volatility REAL NOT NULL,
                regime TEXT NOT NULL,
                UNIQUE(date)
            )
        ''')
        
        # Factor scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                normalized_score REAL NOT NULL,
                current_value REAL,
                weight REAL NOT NULL,
                contribution_pct REAL NOT NULL,
                direction TEXT NOT NULL,
                UNIQUE(date, factor_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_bias_reading(self, results):
        """
        Save a bias reading to the database.
        
        Args:
            results (dict): Results from MacroBiasEngine.run_analysis()
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get date (just the date part, not time)
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp = results['timestamp']
        summary = results['summary']
        
        try:
            # Insert or replace main bias reading
            cursor.execute('''
                INSERT OR REPLACE INTO bias_history 
                (timestamp, date, overall_bias, bias_strength, bias_confidence, 
                 volatility, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                date_str,
                summary['overall_bias'],
                summary['bias_strength_pct'],
                summary['bias_confidence_pct'],
                summary['volatility_pct'],
                summary['regime']
            ))
            
            # Insert or replace factor scores
            for factor in results['factor_scores']:
                cursor.execute('''
                    INSERT OR REPLACE INTO factor_history
                    (date, factor_name, normalized_score, current_value, 
                     weight, contribution_pct, direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date_str,
                    factor['Factor'],
                    factor['Normalized_Score'],
                    factor['Current_Value'],
                    factor['Weight'],
                    factor['Contribution_Pct'],
                    factor['Direction']
                ))
            
            conn.commit()
            
            # Auto-cleanup: Keep only last 30 days
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            cursor.execute('DELETE FROM bias_history WHERE date < ?', (cutoff_date,))
            cursor.execute('DELETE FROM factor_history WHERE date < ?', (cutoff_date,))
            conn.commit()
            
            print(f"✅ Saved bias reading for {date_str}")
            
        except Exception as e:
            print(f"❌ Error saving bias reading: {e}")
            conn.rollback()
        
        finally:
            conn.close()
    
    def get_history(self, days=30):
        """
        Get bias history for the specified number of days.
        
        Args:
            days (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Historical bias data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT * FROM bias_history
            ORDER BY date DESC
            LIMIT {days}
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 0:
            # Reverse to chronological order
            df = df.iloc[::-1].reset_index(drop=True)
        
        return df
    
    def get_factor_history(self, factor_name, days=30):
        """
        Get history for a specific factor.
        
        Args:
            factor_name (str): Name of the factor
            days (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Factor history
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
            SELECT * FROM factor_history
            WHERE factor_name = ?
            ORDER BY date DESC
            LIMIT {days}
        '''
        
        df = pd.read_sql_query(query, conn, params=(factor_name,))
        conn.close()
        
        if len(df) > 0:
            df = df.iloc[::-1].reset_index(drop=True)
        
        return df
    
    def get_date_range(self, start_date, end_date):
        """
        Get bias history for a specific date range.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Historical bias data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM bias_history
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df
    
    def get_specific_date(self, date):
        """
        Get bias reading for a specific date.
        
        Args:
            date (str or datetime): Date to retrieve (YYYY-MM-DD)
            
        Returns:
            dict: Bias data for that date, or None if not found
        """
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get main bias data
        cursor.execute('''
            SELECT * FROM bias_history WHERE date = ?
        ''', (date,))
        
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            return None
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        bias_data = dict(zip(columns, row))
        
        # Get factor data for this date
        cursor.execute('''
            SELECT * FROM factor_history WHERE date = ?
        ''', (date,))
        
        factors = cursor.fetchall()
        factor_columns = [description[0] for description in cursor.description]
        
        bias_data['factors'] = [
            dict(zip(factor_columns, factor)) for factor in factors
        ]
        
        conn.close()
        return bias_data
    
    def get_statistics(self, days=30):
        """
        Get summary statistics for the specified period.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Statistics including accuracy metrics
        """
        df = self.get_history(days)
        
        if len(df) == 0:
            return {
                'error': 'No historical data available',
                'days_analyzed': 0
            }
        
        stats = {
            'days_analyzed': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'bias_distribution': {
                'bullish_days': int((df['overall_bias'] == 'Bullish').sum()),
                'bearish_days': int((df['overall_bias'] == 'Bearish').sum()),
                'neutral_days': int((df['overall_bias'] == 'Neutral').sum())
            },
            'average_metrics': {
                'bias_strength': float(df['bias_strength'].mean()),
                'confidence': float(df['bias_confidence'].mean()),
                'volatility': float(df['volatility'].mean())
            },
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'strongest_signals': {
                'most_bullish': {
                    'date': df.loc[df['bias_strength'].idxmax(), 'date'],
                    'strength': float(df['bias_strength'].max())
                },
                'most_bearish': {
                    'date': df.loc[df['bias_strength'].idxmin(), 'date'],
                    'strength': float(df['bias_strength'].min())
                }
            }
        }
        
        return stats
    
    def export_to_csv(self, filename='bias_history_export.csv', days=None):
        """
        Export history to CSV file.
        
        Args:
            filename (str): Output filename
            days (int): Number of days to export (None = all)
        """
        if days:
            df = self.get_history(days)
        else:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM bias_history ORDER BY date', conn)
            conn.close()
        
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(df)} records to {filename}")
    
    def clear_history(self, confirm=False):
        """
        Clear all historical data (use with caution!).
        
        Args:
            confirm (bool): Must be True to actually delete
        """
        if not confirm:
            print("⚠️  To clear history, call with confirm=True")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM bias_history')
        cursor.execute('DELETE FROM factor_history')
        
        conn.commit()
        conn.close()
        
        print("✅ History cleared")
    
    def get_comparison_with_market(self, market_data_df):
        """
        Compare bias signals with actual market moves.
        
        Args:
            market_data_df (pd.DataFrame): Market data with columns: date, close
            
        Returns:
            pd.DataFrame: Comparison of signals vs actual moves
        """
        bias_df = self.get_history(days=len(market_data_df))
        
        if len(bias_df) == 0:
            return None
        
        # Merge bias with market data
        comparison = pd.merge(
            bias_df[['date', 'overall_bias', 'bias_strength', 'bias_confidence']],
            market_data_df[['date', 'close']],
            on='date',
            how='inner'
        )
        
        # Calculate next day's return
        comparison['next_day_return'] = comparison['close'].pct_change().shift(-1) * 100
        
        # Determine if signal was correct
        comparison['signal_correct'] = (
            ((comparison['bias_strength'] > 15) & (comparison['next_day_return'] > 0)) |
            ((comparison['bias_strength'] < -15) & (comparison['next_day_return'] < 0)) |
            ((comparison['bias_strength'].abs() <= 15) & (comparison['next_day_return'].abs() < 0.5))
        )
        
        # Calculate accuracy
        accuracy = comparison['signal_correct'].mean() * 100
        
        print(f"\n📊 Signal Accuracy: {accuracy:.1f}%")
        print(f"Correct signals: {comparison['signal_correct'].sum()}/{len(comparison)}")
        
        return comparison
    
    def backfill_history(self, days=30, custom_weights=None):
        """
        Backfill historical bias data for past dates.
        This runs the analysis for each past date to build history.
        
        Args:
            days (int): Number of days back to fill (default: 30)
            custom_weights (dict): Optional custom weights for analysis
            
        Returns:
            int: Number of days successfully backfilled
        """
        from macro_bias_engine import MacroBiasEngine
        
        print(f"\n🔄 Backfilling {days} days of historical data...")
        print("This will take a few minutes as we fetch data for each date.\n")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        successful = 0
        failed = 0
        skipped = 0
        
        for days_ago in range(days, -1, -1):
            target_date = datetime.now() - timedelta(days=days_ago)
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Skip if we already have data for this date
            cursor.execute('SELECT date FROM bias_history WHERE date = ?', (date_str,))
            if cursor.fetchone():
                skipped += 1
                print(f"⏭️  {date_str}: Already have data, skipping")
                continue
            
            # Skip weekends
            if target_date.weekday() >= 5:  # Saturday=5, Sunday=6
                skipped += 1
                print(f"⏭️  {date_str}: Weekend, skipping")
                continue
            
            try:
                print(f"📊 {date_str}: Fetching data...", end=" ")
                
                # Create engine with lookback that ends on target date
                # We need to simulate what the data would have looked like on that date
                engine = MacroBiasEngine(weights=custom_weights)
                engine.fetcher.end_date = target_date
                engine.fetcher.start_date = target_date - timedelta(days=engine.fetcher.lookback_days)
                
                # Fetch data as of that date
                factors_data = engine.fetcher.get_all_factors()
                
                # Score factors
                scores_df = engine.scorer.score_all_factors(factors_data, engine.weights)
                
                if len(scores_df) == 0:
                    print("❌ No data available")
                    failed += 1
                    continue
                
                # Calculate bias metrics
                bias_metrics = engine.calculate_bias_metrics(scores_df)
                volatility_metrics = engine.calculate_volatility_regime(factors_data, scores_df)
                scores_df = engine.calculate_factor_contributions(scores_df)
                
                # Create results structure
                results = {
                    'timestamp': target_date.isoformat(),
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
                
                # Save with specific date
                cursor.execute('''
                    INSERT OR REPLACE INTO bias_history 
                    (timestamp, date, overall_bias, bias_strength, bias_confidence, 
                     volatility, regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    results['timestamp'],
                    date_str,
                    results['summary']['overall_bias'],
                    results['summary']['bias_strength_pct'],
                    results['summary']['bias_confidence_pct'],
                    results['summary']['volatility_pct'],
                    results['summary']['regime']
                ))
                
                # Save factor scores
                for factor in results['factor_scores']:
                    cursor.execute('''
                        INSERT OR REPLACE INTO factor_history
                        (date, factor_name, normalized_score, current_value, 
                         weight, contribution_pct, direction)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date_str,
                        factor['Factor'],
                        factor['Normalized_Score'],
                        factor['Current_Value'],
                        factor['Weight'],
                        factor['Contribution_Pct'],
                        factor['Direction']
                    ))
                
                conn.commit()
                print(f"✅ {results['summary']['overall_bias']}")
                successful += 1
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                failed += 1
        
        conn.close()
        
        print(f"\n{'='*70}")
        print(f"✅ Backfill complete!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Skipped: {skipped}")
        print(f"{'='*70}\n")
        
        return successful


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def view_recent_history(days=7):
    """Quick function to view recent bias history."""
    tracker = HistoricalBiasTracker()
    history = tracker.get_history(days)
    
    if len(history) == 0:
        print("📭 No historical data available yet")
        return
    
    print(f"\n📊 BIAS HISTORY - Last {days} Days")
    print("=" * 100)
    
    for _, row in history.iterrows():
        bias_icon = "🟢" if row['overall_bias'] == 'Bullish' else "🔴" if row['overall_bias'] == 'Bearish' else "⚪"
        
        print(f"{row['date']} | {bias_icon} {row['overall_bias']:8s} | "
              f"Strength: {row['bias_strength']:+6.2f}% | "
              f"Confidence: {row['bias_confidence']:5.1f}% | "
              f"Vol: {row['volatility']:5.1f}% ({row['regime']})")
    
    print("=" * 100)


def compare_with_nasdaq(days=30):
    """
    Compare bias signals with Nasdaq performance.
    Requires yfinance to be installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return
    
    # Download Nasdaq data
    print("📥 Downloading Nasdaq data...")
    qqq = yf.download('QQQ', period=f'{days}d', progress=False)
    
    if len(qqq) == 0:
        print("❌ Failed to download Nasdaq data")
        return
    
    # Prepare market data
    market_df = pd.DataFrame({
        'date': qqq.index.strftime('%Y-%m-%d'),
        'close': qqq['Close'].values
    })
    
    # Compare
    tracker = HistoricalBiasTracker()
    comparison = tracker.get_comparison_with_market(market_df)
    
    if comparison is not None:
        print("\n📈 RECENT SIGNALS VS NASDAQ:")
        print(comparison[['date', 'overall_bias', 'bias_strength', 'next_day_return', 'signal_correct']].tail(10))


def backfill_last_month(custom_weights=None):
    """
    Quick function to backfill the last 30 days of historical data.
    
    Args:
        custom_weights (dict): Optional custom weights
    """
    tracker = HistoricalBiasTracker()
    tracker.backfill_history(days=30, custom_weights=custom_weights)
    
    print("\n📊 Historical data is now available!")
    print("Run view_recent_history() to see it.")


def initial_setup():
    """
    Run this once to set up historical tracking with backfilled data.
    """
    print("\n" + "="*70)
    print("🚀 INITIAL SETUP - MACRO BIAS HISTORICAL TRACKING")
    print("="*70)
    
    print("\n📊 This will:")
    print("   1. Create the database")
    print("   2. Backfill the last 30 days of data")
    print("   3. Set you up with a full month of history")
    print("\n⏱️  This will take 3-5 minutes...")
    
    input("\nPress Enter to continue (or Ctrl+C to cancel)...")
    
    # Initialize tracker (creates database)
    tracker = HistoricalBiasTracker()
    
    # Backfill history
    tracker.backfill_history(days=30)
    
    # Show results
    print("\n✅ Setup complete! Here's what you have:\n")
    view_recent_history(days=7)
    
    print("\n💡 Next steps:")
    print("   • Run your Streamlit app to see historical data")
    print("   • Use python view_history.py to explore")
    print("   • Data will auto-update daily from now on")
    print("\n" + "="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HISTORICAL BIAS TRACKER - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize tracker
    tracker = HistoricalBiasTracker()
    
    # Example: View recent history
    print("\n### Example 1: View Recent History ###")
    view_recent_history(days=7)
    
    # Example: Get statistics
    print("\n### Example 2: Get Statistics ###")
    stats = tracker.get_statistics(days=30)
    print(json.dumps(stats, indent=2))
    
    # Example: Get specific date
    print("\n### Example 3: Get Specific Date ###")
    today = datetime.now().strftime('%Y-%m-%d')
    data = tracker.get_specific_date(today)
    if data:
        print(f"Bias for {today}: {data['overall_bias']}")
    else:
        print(f"No data for {today}")
    
    print("\n" + "="*70)
    print("✅ TRACKER EXAMPLES COMPLETE")
    print("="*70)
