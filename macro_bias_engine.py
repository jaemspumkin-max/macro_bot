"""
Macro Bias Engine - 4 DATA SOURCES VERSION
==========================================

Fetches 13 factors from 4 independent sources for maximum reliability:

SOURCE 1: FRED (Federal Reserve Economic Data)
SOURCE 2: Yahoo Finance  
SOURCE 3: Alpha Vantage (backup/alternative)
SOURCE 4: Treasury Direct (US Treasury official)

Author: Trading Analytics
Version: 5.0 - Multi-Source Complete
Date: 2026-02-15
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
from io import StringIO
import time
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from pandas_datareader import data as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False

# ============================================================================
# WEIGHTS
# ============================================================================

NQ_FUTURES_WEIGHTS = {
    'VIX_Index': 3.5,
    'NQ_SPY_Ratio': 3.0,
    'Semiconductor_Index': 2.5,
    'SPY_Momentum': 2.5,
    'Put_Call_Ratio': 2.5,
    '10Y_Treasury_Yield': 2.0,
    'High_Yield_Spreads': 2.0,
    'Treasury_Curve': 1.5,
    'DXY_Dollar_Index': 1.5,
    'Oil_Prices': 1.0,
    'Gold_SPY_Ratio': 1.0,
    'Credit_Spreads_BAA_AAA': 0.5,
    'M2_Money_Supply': 0.1,
}

DEFAULT_FACTOR_WEIGHTS = NQ_FUTURES_WEIGHTS

FRED_SERIES = {
    '10Y_Treasury_Yield': 'DGS10',
    '2Y_Treasury_Yield': 'DGS2',
    'M2_Money_Supply': 'M2SL',
    'BAA_Yield': 'DBAA',
    'AAA_Yield': 'DAAA',
}

ALPHA_VANTAGE_KEY = "demo"  # Get free key at https://www.alphavantage.co/

VOLATILITY_THRESHOLDS = {'Low': (0, 33), 'Medium': (34, 66), 'High': (67, 100)}

# ============================================================================
# HELPERS
# ============================================================================

def is_valid_series(series):
    if series is None or not isinstance(series, pd.Series) or len(series) == 0:
        return False
    try:
        return not series.empty
    except:
        return False

# ============================================================================
# MULTI-SOURCE FETCHER
# ============================================================================

class MultiSourceFetcher:
    
    def __init__(self, lookback_days=90, alpha_key=None):
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.alpha_key = alpha_key or ALPHA_VANTAGE_KEY
        self.stats = {'fred': 0, 'yahoo': 0, 'alpha': 0, 'treasury': 0}
    
    def log(self, msg):
        print(f"  {msg}")
    
    # SOURCE 1: FRED
    def fetch_fred_csv(self, series_id):
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
            params = {
                'id': series_id,
                'cosd': (self.start_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'coed': self.end_date.strftime('%Y-%m-%d'),
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(StringIO(r.text))
                if len(df) >= 2:
                    df.columns = ['DATE', 'VALUE']
                    df['DATE'] = pd.to_datetime(df['DATE'])
                    series = pd.to_numeric(df.set_index('DATE')['VALUE'], errors='coerce')
                    series = series.ffill().dropna()[self.start_date:]
                    if len(series) > 0:
                        self.stats['fred'] += 1
                        return series
        except:
            pass
        return None
    
    # SOURCE 2: YAHOO
    def fetch_yahoo(self, ticker):
        if not YFINANCE_AVAILABLE:
            return None
        try:
            data = yf.download(ticker, start=self.start_date - timedelta(days=10),
                             end=self.end_date, progress=False, show_errors=False)
            if not data.empty:
                series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
                series = series.ffill().dropna()[self.start_date:]
                if len(series) > 0:
                    self.stats['yahoo'] += 1
                    return series
        except:
            pass
        return None
    
    # SOURCE 3: ALPHA VANTAGE
    def fetch_alpha_stock(self, ticker):
        try:
            clean = ticker.replace('^', '').replace('-Y.NYB', '')
            url = "https://www.alphavantage.co/query"
            r = requests.get(url, params={
                'function': 'TIME_SERIES_DAILY',
                'symbol': clean,
                'apikey': self.alpha_key,
                'outputsize': 'compact'
            }, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if 'Time Series (Daily)' in data:
                    ts = data['Time Series (Daily)']
                    dates = [pd.to_datetime(d) for d in ts.keys()]
                    closes = [float(ts[d]['4. close']) for d in ts.keys()]
                    series = pd.Series(closes, index=dates).sort_index()
                    series = series.ffill().dropna()[self.start_date:]
                    if len(series) > 0:
                        self.stats['alpha'] += 1
                        time.sleep(0.3)  # Rate limit
                        return series
        except:
            pass
        return None
    
    # SOURCE 4: TREASURY DIRECT
    def fetch_treasury_direct(self):
        try:
            year = datetime.now().year
            url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{year}?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(StringIO(r.text))
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                
                y10 = pd.to_numeric(df.get('10 Yr'), errors='coerce')
                y10 = y10.ffill().dropna()[self.start_date:]
                
                y2 = pd.to_numeric(df.get('2 Yr'), errors='coerce')
                y2 = y2.ffill().dropna()[self.start_date:] if '2 Yr' in df.columns else None
                
                if len(y10) > 0:
                    self.stats['treasury'] += 1
                    return {'10Y': y10, '2Y': y2}
        except:
            pass
        return None
    
    # SMART MULTI-SOURCE GETTERS
    def get_treasury_smart(self, series_id, maturity='10Y'):
        self.log(f"Fetching {maturity} from 3 sources...")
        
        # Try Treasury Direct first (official)
        td = self.fetch_treasury_direct()
        if td and maturity in td and is_valid_series(td[maturity]):
            return td[maturity]
        
        # Try FRED
        s = self.fetch_fred_csv(series_id)
        if is_valid_series(s):
            return s
        
        return None
    
    def get_stock_smart(self, ticker):
        self.log(f"Fetching {ticker} from 3 sources...")
        
        # Try Yahoo first
        s = self.fetch_yahoo(ticker)
        if is_valid_series(s):
            return s
        
        # Try Alpha Vantage
        s = self.fetch_alpha_stock(ticker)
        if is_valid_series(s):
            return s
        
        return None
    
    def calc_ratio(self, s1, s2):
        if is_valid_series(s1) and is_valid_series(s2):
            df = pd.DataFrame({'A': s1, 'B': s2}).dropna()
            return (df['A'] / df['B']) * 100 if len(df) > 0 else None
        return None
    
    # FACTORS
    def get_all_factors(self):
        print("\n" + "="*80)
        print("🌍 4-SOURCE ENGINE | FRED | Yahoo | Alpha Vantage | Treasury Direct")
        print("="*80)
        
        factors = {}
        
        self.log("\n[1/13] 10Y Treasury")
        factors['10Y_Treasury_Yield'] = self.get_treasury_smart(FRED_SERIES['10Y_Treasury_Yield'], '10Y')
        
        self.log("\n[2/13] DXY")
        factors['DXY_Dollar_Index'] = self.get_stock_smart('DX-Y.NYB')
        
        self.log("\n[3/13] M2")
        factors['M2_Money_Supply'] = self.fetch_fred_csv(FRED_SERIES['M2_Money_Supply'])
        
        self.log("\n[4/13] Credit Spreads")
        baa = self.fetch_fred_csv(FRED_SERIES['BAA_Yield'])
        aaa = self.fetch_fred_csv(FRED_SERIES['AAA_Yield'])
        if is_valid_series(baa) and is_valid_series(aaa):
            df = pd.DataFrame({'BAA': baa, 'AAA': aaa}).dropna()
            factors['Credit_Spreads_BAA_AAA'] = df['BAA'] - df['AAA'] if len(df) > 0 else None
        else:
            factors['Credit_Spreads_BAA_AAA'] = None
        
        self.log("\n[5/13] VIX")
        factors['VIX_Index'] = self.get_stock_smart('^VIX')
        
        self.log("\n[6/13] HY Spreads")
        hyg = self.get_stock_smart('HYG')
        agg = self.get_stock_smart('AGG')
        if is_valid_series(hyg) and is_valid_series(agg):
            df = pd.DataFrame({'HYG': hyg, 'AGG': agg}).dropna()
            factors['High_Yield_Spreads'] = (df['HYG'] / df['AGG']).pct_change(20) * 100 if len(df) > 20 else None
        else:
            factors['High_Yield_Spreads'] = None
        
        self.log("\n[7/13] SPY Momentum")
        spy = self.get_stock_smart('SPY')
        if is_valid_series(spy) and len(spy) > 50:
            ma20 = spy.rolling(20).mean()
            ma50 = spy.rolling(50).mean()
            factors['SPY_Momentum'] = ((ma20 / ma50) - 1) * 100
        else:
            factors['SPY_Momentum'] = None
        
        self.log("\n[8/13] Gold/SPY")
        gld = self.get_stock_smart('GLD')
        r = self.calc_ratio(gld, spy)
        factors['Gold_SPY_Ratio'] = r.pct_change(20) * 100 if r is not None and len(r) > 20 else None
        
        self.log("\n[9/13] NQ/SPY ⭐")
        qqq = self.get_stock_smart('QQQ')
        r = self.calc_ratio(qqq, spy)
        factors['NQ_SPY_Ratio'] = r.pct_change(20) * 100 if r is not None and len(r) > 20 else None
        
        self.log("\n[10/13] Yield Curve")
        y10 = self.get_treasury_smart(FRED_SERIES['10Y_Treasury_Yield'], '10Y')
        y2 = self.get_treasury_smart(FRED_SERIES['2Y_Treasury_Yield'], '2Y')
        if is_valid_series(y10) and is_valid_series(y2):
            df = pd.DataFrame({'Y10': y10, 'Y2': y2}).dropna()
            factors['Treasury_Curve'] = df['Y10'] - df['Y2'] if len(df) > 0 else None
        else:
            factors['Treasury_Curve'] = None
        
        self.log("\n[11/13] SMH ⭐")
        smh = self.get_stock_smart('SMH')
        factors['Semiconductor_Index'] = smh.pct_change(20) * 100 if is_valid_series(smh) and len(smh) > 20 else None
        
        self.log("\n[12/13] Put/Call")
        vxn = self.get_stock_smart('^VXN')
        vix = self.get_stock_smart('^VIX')
        if is_valid_series(vxn) and is_valid_series(vix):
            df = pd.DataFrame({'VXN': vxn, 'VIX': vix}).dropna()
            factors['Put_Call_Ratio'] = (df['VXN'] / df['VIX'] - 1) * 100 if len(df) > 0 else None
        else:
            factors['Put_Call_Ratio'] = None
        
        self.log("\n[13/13] Oil")
        uso = self.get_stock_smart('USO')
        factors['Oil_Prices'] = uso.pct_change(20) * 100 if is_valid_series(uso) and len(uso) > 20 else None
        
        valid = sum(1 for v in factors.values() if is_valid_series(v))
        print(f"\n{'='*80}")
        print(f"✅ {valid}/13 FACTORS | Sources: FRED={self.stats['fred']} Yahoo={self.stats['yahoo']} Alpha={self.stats['alpha']} Treasury={self.stats['treasury']}")
        print(f"{'='*80}\n")
        
        return factors

# ============================================================================
# SCORER
# ============================================================================

class FactorScorer:
    
    @staticmethod
    def calc_change(s, p=20):
        if not is_valid_series(s) or len(s) < p:
            return 0.0
        try:
            c, pa = float(s.iloc[-1]), float(s.iloc[-p])
            return (c - pa) / pa * 100 if not pd.isna(c) and not pd.isna(pa) and pa != 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def calc_z(s, w=60):
        if not is_valid_series(s):
            return 0.0
        w = min(w, len(s)) if len(s) < w else w
        try:
            r = s.iloc[-w:]
            m, sd = float(r.mean()), float(r.std())
            c = float(s.iloc[-1])
            return (c - m) / sd if sd > 0 and not pd.isna(c) else 0.0
        except:
            return 0.0
    
    def score(self, s, name, inv=False):
        if not is_valid_series(s):
            return 0.0, 0.0, 0.0, None
        try:
            p = self.calc_change(s)
            z = self.calc_z(s)
            c = float(s.iloc[-1])
            n = np.tanh(z / 2)
            n = -n if inv else n
            return float(np.clip(n, -1, 1)), float(p), float(z), float(c)
        except:
            return 0.0, 0.0, 0.0, None
    
    def score_all(self, data, weights):
        results = []
        bearish = ['10Y_Treasury_Yield', 'DXY_Dollar_Index', 'Credit_Spreads_BAA_AAA',
                  'VIX_Index', 'High_Yield_Spreads', 'Gold_SPY_Ratio', 'Put_Call_Ratio', 'Oil_Prices']
        
        for name, s in data.items():
            if not is_valid_series(s):
                continue
            sc, p, z, c = self.score(s, name, name in bearish)
            w = weights.get(name, 1.0)
            results.append({
                'Factor': name,
                'Current_Value': c,
                'Percent_Change_20D': p,
                'Z_Score': z,
                'Normalized_Score': sc,
                'Weight': w,
                'Weighted_Score': sc * w,
                'Direction': 'Bullish' if sc > 0 else 'Bearish' if sc < 0 else 'Neutral'
            })
        
        return pd.DataFrame(results) if results else pd.DataFrame()

# ============================================================================
# ENGINE
# ============================================================================

class MacroBiasEngine:
    
    def __init__(self, weights=None, alpha_key=None):
        self.weights = weights or NQ_FUTURES_WEIGHTS
        self.fetcher = MultiSourceFetcher(alpha_key=alpha_key)
        self.scorer = FactorScorer()
    
    def calc_metrics(self, df):
        if len(df) == 0:
            return {'overall_bias': 'Neutral', 'bias_strength': 0.0, 'bias_confidence': 0.0}
        
        tw = float(df['Weighted_Score'].sum())
        mp = float(df['Weight'].sum())
        
        if mp == 0:
            return {'overall_bias': 'Neutral', 'bias_strength': 0.0, 'bias_confidence': 0.0}
        
        st = (tw / mp) * 100
        co = (abs(tw) / mp) * 100
        
        return {
            'overall_bias': 'Bullish' if st > 15 else 'Bearish' if st < -15 else 'Neutral',
            'bias_strength': round(st, 2),
            'bias_confidence': round(co, 2),
        }
    
    def calc_contribs(self, df):
        if len(df) == 0:
            return df
        ta = float(df['Weighted_Score'].abs().sum())
        df['Contribution_Pct'] = (df['Weighted_Score'].abs() / ta * 100).round(2) if ta > 0 else 0
        return df
    
    def calc_vol(self, data, df):
        vc = []
        if 'VIX_Index' in data and is_valid_series(data['VIX_Index']):
            try:
                vc.append(min(float(data['VIX_Index'].iloc[-1]) / 50 * 100, 100))
            except:
                pass
        
        vp = float(np.mean(vc)) if vc else 50.0
        return {
            'volatility_pct': round(vp, 2),
            'regime': 'Low Volatility' if vp <= 33 else 'Medium Volatility' if vp <= 66 else 'High Volatility'
        }
    
    def run_analysis(self):
        data = self.fetcher.get_all_factors()
        df = self.scorer.score_all(data, self.weights)
        bm = self.calc_metrics(df)
        vm = self.calc_vol(data, df)
        df = self.calc_contribs(df)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'bias_metrics': bm,
            'volatility_metrics': vm,
            'factor_scores': df.to_dict('records'),
            'summary': {
                'overall_bias': bm['overall_bias'],
                'bias_strength_pct': bm['bias_strength'],
                'bias_confidence_pct': bm['bias_confidence'],
                'volatility_pct': vm['volatility_pct'],
                'regime': vm['regime']
            },
            'sources_used': self.fetcher.stats
        }

# ============================================================================
# OUTPUT
# ============================================================================

class OutputFormatter:
    
    @staticmethod
    def print_summary(r):
        s = r['summary']
        print(f"\n{'='*70}\n📊 {s['overall_bias']} | Strength: {s['bias_strength_pct']:+.2f}% | Conf: {s['bias_confidence_pct']:.2f}%")
        print(f"🌊 {s['regime']} | Vol: {s['volatility_pct']:.2f}%")
        
        if 'sources_used' in r:
            u = r['sources_used']
            print(f"\n🌍 Sources: FRED={u.get('fred',0)} Yahoo={u.get('yahoo',0)} Alpha={u.get('alpha',0)} Treasury={u.get('treasury',0)}")
        
        print("\n📈 TOP FACTORS:")
        df = pd.DataFrame(r['factor_scores'])
        if len(df) > 0:
            for _, row in df.sort_values('Contribution_Pct', ascending=False).head(5).iterrows():
                i = "🟢" if row['Direction'] == 'Bullish' else "🔴" if row['Direction'] == 'Bearish' else "⚪"
                print(f"{i} {row['Factor']:30s} | {row['Normalized_Score']:+.2f} | {row['Contribution_Pct']:.1f}%")
        print(f"{'='*70}\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 4-SOURCE NQ ENGINE")
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    OutputFormatter.print_summary(results)
    print("💡 Bias >+20% + Conf >70% = LONG | Bias <-20% + Conf >70% = SHORT\n")
