"""
Macro Bias Engine - Test Suite
==============================

This script tests all core functionality of the macro bias engine
to ensure everything is working correctly.

Run with: python test_engine.py
"""

import sys
from datetime import datetime

def test_imports():
    """Test 1: Check if all required imports work."""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    errors = []
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        errors.append(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        errors.append(f"❌ numpy import failed: {e}")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        errors.append(f"⚠️  yfinance import failed: {e}")
        errors.append("   Install with: pip install yfinance")
    
    try:
        from pandas_datareader import data as pdr
        print("✅ pandas_datareader imported successfully")
    except ImportError as e:
        errors.append(f"⚠️  pandas_datareader import failed: {e}")
        errors.append("   Install with: pip install pandas-datareader")
    
    try:
        from macro_bias_engine import MacroBiasEngine, OutputFormatter
        print("✅ macro_bias_engine imported successfully")
    except ImportError as e:
        errors.append(f"❌ macro_bias_engine import failed: {e}")
        errors.append("   Make sure macro_bias_engine.py is in the same directory")
    
    if errors:
        print("\n⚠️  Import Issues:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n✅ All imports successful!")
        return True


def test_engine_initialization():
    """Test 2: Check if engine initializes correctly."""
    print("\n" + "="*70)
    print("TEST 2: Engine Initialization")
    print("="*70)
    
    try:
        from macro_bias_engine import MacroBiasEngine
        
        # Test default initialization
        engine = MacroBiasEngine()
        print("✅ Default initialization successful")
        
        # Test custom weights
        custom_weights = {
            '10Y_Treasury_Yield': 2.5,
            'DXY_Dollar_Index': 1.5,
            'M2_Money_Supply': 2.0,
            'Credit_Spreads_BAA_AAA': 2.0,
            'VIX_Index': 1.0,
        }
        engine_custom = MacroBiasEngine(weights=custom_weights)
        print("✅ Custom weights initialization successful")
        
        return True
    
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_data_fetching():
    """Test 3: Check if data fetching works."""
    print("\n" + "="*70)
    print("TEST 3: Data Fetching")
    print("="*70)
    
    try:
        from macro_bias_engine import MacroDataFetcher
        
        fetcher = MacroDataFetcher(lookback_days=30)  # Short lookback for faster test
        
        print("Attempting to fetch data (this may take 30-60 seconds)...")
        factors_data = fetcher.get_all_factors()
        
        # Check which factors were successfully fetched
        success_count = 0
        for factor_name, data in factors_data.items():
            if data is not None and len(data) > 0:
                print(f"✅ {factor_name}: Fetched {len(data)} data points")
                success_count += 1
            else:
                print(f"⚠️  {factor_name}: No data fetched (might be expected)")
        
        if success_count >= 3:  # At least 3 factors should work
            print(f"\n✅ Data fetching successful ({success_count}/6 factors)")
            return True
        else:
            print(f"\n⚠️  Limited data fetched ({success_count}/6 factors)")
            print("   This might be due to network issues or API availability")
            return False
    
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factor_scoring():
    """Test 4: Check if factor scoring works."""
    print("\n" + "="*70)
    print("TEST 4: Factor Scoring")
    print("="*70)
    
    try:
        from macro_bias_engine import FactorScorer
        import pandas as pd
        import numpy as np
        
        # Create synthetic test data
        test_data = pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=pd.date_range('2024-01-01', periods=100, freq='D')
        )
        
        scorer = FactorScorer()
        
        # Test scoring
        score, pct_change, z_score, current_value = scorer.score_factor(
            test_data, 'Test_Factor', invert=False
        )
        
        print(f"✅ Factor scoring successful")
        print(f"   Score: {score:.3f}")
        print(f"   Percent Change: {pct_change:.2f}%")
        print(f"   Z-Score: {z_score:.3f}")
        print(f"   Current Value: {current_value:.2f}")
        
        # Validate score is in valid range
        if -1 <= score <= 1:
            print(f"✅ Score is within valid range [-1, 1]")
            return True
        else:
            print(f"❌ Score {score} is outside valid range [-1, 1]")
            return False
    
    except Exception as e:
        print(f"❌ Factor scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_analysis():
    """Test 5: Run complete analysis."""
    print("\n" + "="*70)
    print("TEST 5: Full Analysis")
    print("="*70)
    
    try:
        from macro_bias_engine import MacroBiasEngine, OutputFormatter
        
        print("Running full macro bias analysis...")
        print("(This may take 1-2 minutes to fetch all data)\n")
        
        engine = MacroBiasEngine()
        results = engine.run_analysis()
        
        # Validate results structure
        required_keys = ['timestamp', 'bias_metrics', 'volatility_metrics', 
                        'factor_scores', 'summary']
        
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            print(f"❌ Missing keys in results: {missing_keys}")
            return False
        
        print("✅ All required result keys present")
        
        # Validate summary
        summary = results['summary']
        required_summary_keys = ['overall_bias', 'bias_strength_pct', 
                                'bias_confidence_pct', 'volatility_pct', 'regime']
        
        missing_summary_keys = [key for key in required_summary_keys 
                               if key not in summary]
        
        if missing_summary_keys:
            print(f"❌ Missing keys in summary: {missing_summary_keys}")
            return False
        
        print("✅ All summary keys present")
        
        # Validate value ranges
        if not -100 <= summary['bias_strength_pct'] <= 100:
            print(f"❌ Invalid bias_strength: {summary['bias_strength_pct']}")
            return False
        
        if not 0 <= summary['bias_confidence_pct'] <= 100:
            print(f"❌ Invalid confidence: {summary['bias_confidence_pct']}")
            return False
        
        if summary['overall_bias'] not in ['Bullish', 'Bearish', 'Neutral']:
            print(f"❌ Invalid overall_bias: {summary['overall_bias']}")
            return False
        
        print("✅ All values within valid ranges")
        
        # Print results
        print("\n" + "-"*70)
        print("ANALYSIS RESULTS:")
        print("-"*70)
        OutputFormatter.print_summary(results)
        
        return True
    
    except Exception as e:
        print(f"❌ Full analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_formats():
    """Test 6: Test all output formats."""
    print("\n" + "="*70)
    print("TEST 6: Output Formats")
    print("="*70)
    
    try:
        from macro_bias_engine import MacroBiasEngine, OutputFormatter
        
        # Run analysis
        engine = MacroBiasEngine()
        results = engine.run_analysis()
        
        # Test JSON output
        try:
            json_output = OutputFormatter.to_json(results)
            print("✅ JSON export successful")
            print(f"   Length: {len(json_output)} characters")
        except Exception as e:
            print(f"❌ JSON export failed: {e}")
            return False
        
        # Test DataFrame output
        try:
            df = OutputFormatter.to_dataframe(results)
            print("✅ DataFrame export successful")
            print(f"   Shape: {df.shape}")
        except Exception as e:
            print(f"❌ DataFrame export failed: {e}")
            return False
        
        # Test dashboard dict output
        try:
            dashboard_data = OutputFormatter.to_dashboard_dict(results)
            print("✅ Dashboard dict export successful")
            print(f"   Keys: {list(dashboard_data.keys())}")
        except Exception as e:
            print(f"❌ Dashboard dict export failed: {e}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Output format tests failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*70)
    print("MACRO BIAS ENGINE - TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_engine_initialization),
        ("Data Fetching Test", test_data_fetching),
        ("Factor Scoring Test", test_factor_scoring),
        ("Full Analysis Test", test_full_analysis),
        ("Output Format Test", test_output_formats),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The engine is working correctly.")
        print("\nNext steps:")
        print("  1. Try running: python usage_examples.py")
        print("  2. Create visualizations: python visualizations.py")
        print("  3. Start the API: python flask_api.py")
        print("  4. Open the dashboard: dashboard.html")
    elif passed >= total * 0.7:
        print("\n⚠️  MOST TESTS PASSED! Engine is partially working.")
        print("\nSome features may be limited due to:")
        print("  • Network connectivity issues")
        print("  • Missing optional dependencies")
        print("  • API availability")
        print("\nThe engine should still be usable for most purposes.")
    else:
        print("\n❌ MANY TESTS FAILED! Please check:")
        print("  1. All required packages are installed")
        print("  2. You have internet connectivity")
        print("  3. macro_bias_engine.py is in the same directory")
        print("\nInstall missing packages with:")
        print("  pip install pandas numpy yfinance pandas-datareader")
    
    print("\n" + "="*70)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
