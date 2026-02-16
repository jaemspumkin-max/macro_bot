"""
Macro Bias Engine - Streamlit Dashboard (Updated for 13 Factors)
================================================================

Complete dashboard with all 13 factors, 4 data sources, and NQ-optimized interface.

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Import engine
try:
    from macro_bias_engine import MacroBiasEngine, OutputFormatter
    ENGINE_AVAILABLE = True
except ImportError:
    st.error("❌ macro_bias_engine.py not found!")
    st.stop()

try:
    from historical_tracker import HistoricalBiasTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    # Create dummy class
    class HistoricalBiasTracker:
        def save_bias_reading(self, results): pass
        def get_history(self, days): return pd.DataFrame()
        def backfill_history(self, days, custom_weights=None): return 0

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="NQ Macro Bias Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {padding: 0rem 1rem;}
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
h1 {color: #1f77b4; border-bottom: 3px solid #1f77b4;}
.bullish {color: #00c853; font-weight: bold;}
.bearish {color: #ff1744; font-weight: bold;}
.neutral {color: #ffa726; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE
# ============================================================================

@st.cache_data(ttl=900)
def run_analysis(weights=None, alpha_key=None):
    if weights:
        engine = MacroBiasEngine(weights=weights, alpha_key=alpha_key)
    else:
        engine = MacroBiasEngine(alpha_key=alpha_key)
    return engine.run_analysis()

# ============================================================================
# SIDEBAR
# ============================================================================

# Initialize alpha_key with default value
alpha_key = "demo"

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Weight preset selector
    st.markdown("### 📊 Weight Preset")
    preset = st.selectbox(
        "Choose Strategy:",
        ["NQ Futures (5-min) ⭐", "Scalping (Ultra Short)", "Swing Trading", "Custom"],
        help="Optimized weight configurations for different trading styles"
    )
    
    custom_weights = None
    
    if preset == "Custom":
        st.markdown("### 🎛️ Custom Weights")
        st.caption("Adjust factor importance (0-5)")
        
        with st.expander("🔥 **HIGH PRIORITY** (Fast-moving)", expanded=True):
            w_vix = st.slider("VIX Index", 0.0, 5.0, 3.5, 0.5, key="vix")
            w_nq_spy = st.slider("NQ/SPY Ratio ⭐", 0.0, 5.0, 3.0, 0.5, key="nqspy")
            w_smh = st.slider("Semiconductors (SMH) ⭐", 0.0, 5.0, 2.5, 0.5, key="smh")
            w_spy_mom = st.slider("SPY Momentum", 0.0, 5.0, 2.5, 0.5, key="spymom")
            w_pc = st.slider("Put/Call Ratio", 0.0, 5.0, 2.5, 0.5, key="pc")
        
        with st.expander("⚖️ **MEDIUM PRIORITY**", expanded=False):
            w_10y = st.slider("10Y Treasury", 0.0, 5.0, 2.0, 0.5, key="10y")
            w_hy = st.slider("High Yield Spreads", 0.0, 5.0, 2.0, 0.5, key="hy")
            w_curve = st.slider("Yield Curve", 0.0, 5.0, 1.5, 0.5, key="curve")
            w_dxy = st.slider("DXY Dollar", 0.0, 5.0, 1.5, 0.5, key="dxy")
        
        with st.expander("📉 **LOW PRIORITY** (Slow)", expanded=False):
            w_oil = st.slider("Oil Prices", 0.0, 5.0, 1.0, 0.5, key="oil")
            w_gold = st.slider("Gold/SPY", 0.0, 5.0, 1.0, 0.5, key="gold")
            w_credit = st.slider("Credit Spreads", 0.0, 5.0, 0.5, 0.5, key="credit")
            w_m2 = st.slider("M2 Money Supply", 0.0, 5.0, 0.1, 0.1, key="m2")
        
        custom_weights = {
            'VIX_Index': w_vix,
            'NQ_SPY_Ratio': w_nq_spy,
            'Semiconductor_Index': w_smh,
            'SPY_Momentum': w_spy_mom,
            'Put_Call_Ratio': w_pc,
            '10Y_Treasury_Yield': w_10y,
            'High_Yield_Spreads': w_hy,
            'Treasury_Curve': w_curve,
            'DXY_Dollar_Index': w_dxy,
            'Oil_Prices': w_oil,
            'Gold_SPY_Ratio': w_gold,
            'Credit_Spreads_BAA_AAA': w_credit,
            'M2_Money_Supply': w_m2,
        }
    
    st.markdown("---")
    
    # Alpha Vantage key (optional)
    with st.expander("🔑 Alpha Vantage API Key (Optional)"):
        alpha_key = st.text_input(
            "API Key:",
            value="demo",
            help="Get free key at alphavantage.co - improves reliability"
        )
    
    st.markdown("---")
    
    # Refresh
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    auto_refresh = st.checkbox("⏰ Auto-refresh (5 min)", value=False)
    
    st.markdown("---")
    
    # Historical Data Section
    if TRACKER_AVAILABLE:
        st.markdown("### 📊 Historical Data")
        
        tracker = HistoricalBiasTracker()
        history_df = tracker.get_history(days=30)
        
        if len(history_df) < 5:
            st.warning("📭 Limited historical data")
            if st.button("🔄 Backfill Last 30 Days", help="Fetch historical data for past dates"):
                with st.spinner("⏳ Backfilling history... This takes 3-5 minutes..."):
                    try:
                        successful = tracker.backfill_history(days=30, custom_weights=custom_weights if custom_weights else None)
                        st.success(f"✅ Backfilled {successful} days of data!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error backfilling: {e}")
        
        show_history = st.checkbox("📅 View Past Bias Readings", value=False)
        
        if show_history:
            history_days = st.slider("Days to show", 7, 90, 30)
            
            history_df = tracker.get_history(days=history_days)
            
            if len(history_df) > 0:
                st.caption(f"📈 Showing last {len(history_df)} days")
            else:
                st.caption("📭 No historical data yet")
    
    st.markdown("---")
    
    # Information
    st.markdown("### ℹ️ About")
    st.info("""
    **NQ Futures Engine**
    
    🎯 13 Factors  
    🌍 4 Data Sources  
    ⚡ NQ-Optimized  
    
    Sources:
    - FRED
    - Yahoo Finance
    - Alpha Vantage
    - Treasury Direct
    """)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 NQ Futures Macro Bias Engine")
    st.caption("13 Factors | 4 Data Sources | Real-Time Analysis")
with col2:
    st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")

# Run analysis
try:
    with st.spinner("🔄 Fetching data from 4 sources..."):
        results = run_analysis(custom_weights, alpha_key)
        summary = results['summary']
        
        # Save to history
        if TRACKER_AVAILABLE:
            try:
                tracker = HistoricalBiasTracker()
                tracker.save_bias_reading(results)
            except:
                pass
        
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ============================================================================
# KEY METRICS
# ============================================================================

st.markdown("## 📊 Overall Bias")

col1, col2, col3, col4 = st.columns(4)

with col1:
    bias = summary['overall_bias']
    color = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
    st.metric(
        "Overall Bias",
        f"{color} {bias}",
        help="Market directional bias"
    )

with col2:
    strength = summary['bias_strength_pct']
    st.metric(
        "Bias Strength",
        f"{strength:+.1f}%",
        delta=f"{strength:.1f}%",
        help="Magnitude of bias (-100 to +100)"
    )

with col3:
    confidence = summary['bias_confidence_pct']
    st.metric(
        "Signal Confidence",
        f"{confidence:.1f}%",
        delta=f"{confidence:.1f}%" if confidence > 50 else f"-{100-confidence:.1f}%",
        help="Reliability of signal (0-100%)"
    )

with col4:
    volatility = summary['volatility_pct']
    regime = summary['regime']
    st.metric(
        "Volatility",
        f"{volatility:.1f}%",
        delta=regime.split()[0],
        help=f"Market volatility regime: {regime}"
    )

# Trading signal
st.markdown("---")
st.markdown("## 🎯 Trading Signal")

if strength > 20 and confidence > 70:
    st.success(f"### 🟢 STRONG LONG SIGNAL")
    st.write(f"**Action:** Go long NQ | **Confidence:** {confidence:.0f}% | **Strength:** {strength:+.1f}%")
    st.write("✅ High probability setup - trade with normal size")
elif strength < -20 and confidence > 70:
    st.error(f"### 🔴 STRONG SHORT SIGNAL")
    st.write(f"**Action:** Go short NQ | **Confidence:** {confidence:.0f}% | **Strength:** {strength:+.1f}%")
    st.write("✅ High probability setup - trade with normal size")
elif confidence > 60:
    if strength > 0:
        st.success(f"### 🟢 MODERATE LONG SIGNAL")
        st.write(f"**Action:** Cautious longs | **Confidence:** {confidence:.0f}% | **Strength:** {strength:+.1f}%")
        st.write("⚠️ Moderate signal - reduce size by 50%")
    else:
        st.error(f"### 🔴 MODERATE SHORT SIGNAL")
        st.write(f"**Action:** Cautious shorts | **Confidence:** {confidence:.0f}% | **Strength:** {strength:+.1f}%")
        st.write("⚠️ Moderate signal - reduce size by 50%")
else:
    st.warning(f"### ⚪ NEUTRAL / LOW CONFIDENCE")
    st.write(f"**Action:** Sit out or micro size | **Confidence:** {confidence:.0f}% | **Strength:** {strength:+.1f}%")
    st.write("⚠️ No clear signal - avoid trading or use 25% size")

# ============================================================================
# DATA SOURCES
# ============================================================================

if 'sources_used' in results:
    st.markdown("---")
    st.markdown("## 🌍 Data Sources Used")
    
    sources = results['sources_used']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FRED", sources.get('fred', 0), help="Federal Reserve data")
    with col2:
        st.metric("Yahoo", sources.get('yahoo', 0), help="Market data")
    with col3:
        st.metric("Alpha Vantage", sources.get('alpha', 0), help="Backup data")
    with col4:
        st.metric("Treasury Direct", sources.get('treasury', 0), help="Official bonds")

# ============================================================================
# FACTOR BREAKDOWN
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Factor Breakdown")

# Create dataframe
factor_df = pd.DataFrame(results['factor_scores'])

if len(factor_df) > 0:
    # Sort by contribution
    factor_df = factor_df.sort_values('Contribution_Pct', ascending=False)
    
    # Show table
    display_df = factor_df[['Factor', 'Direction', 'Normalized_Score', 'Weight', 'Contribution_Pct']].copy()
    display_df['Normalized_Score'] = display_df['Normalized_Score'].apply(lambda x: f"{x:+.2f}")
    display_df['Contribution_Pct'] = display_df['Contribution_Pct'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Contribution chart
    st.markdown("### 📊 Factor Contributions")
    
    fig = go.Figure()
    
    # Color by direction
    colors = ['green' if d == 'Bullish' else 'red' if d == 'Bearish' else 'gray' 
              for d in factor_df['Direction']]
    
    fig.add_trace(go.Bar(
        y=factor_df['Factor'],
        x=factor_df['Weighted_Score'],
        orientation='h',
        marker_color=colors,
        text=factor_df['Contribution_Pct'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Weighted Factor Scores",
        xaxis_title="Weighted Score",
        yaxis_title="",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# GAUGES
# ============================================================================

st.markdown("---")
st.markdown("## 🎛️ Visual Indicators")

col1, col2 = st.columns(2)

with col1:
    # Bias strength gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=strength,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Bias Strength", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-100, -15], 'color': '#ffcdd2'},
                {'range': [-15, 15], 'color': '#fff9c4'},
                {'range': [15, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': strength
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Confidence gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Signal Confidence", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgreen" if confidence > 70 else "orange" if confidence > 50 else "red"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcdd2'},
                {'range': [50, 70], 'color': '#fff9c4'},
                {'range': [70, 100], 'color': '#c8e6c9'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EXPORT
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Export Data")

col1, col2 = st.columns(2)

with col1:
    # Export factor scores
    csv = factor_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Factor Scores (CSV)",
        data=csv,
        file_name=f"factor_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    # Export summary
    import json
    json_str = json.dumps(results['summary'], indent=2)
    st.download_button(
        label="📥 Download Summary (JSON)",
        data=json_str,
        file_name=f"bias_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# ============================================================================
# HISTORICAL TRACKING
# ============================================================================

if TRACKER_AVAILABLE:
    st.markdown("---")
    st.markdown("## 📈 Historical Bias Readings")
    
    tracker = HistoricalBiasTracker()
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        history_period = st.selectbox(
            "Time Period:",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
    
    # Get historical data
    history_df = tracker.get_history(days=history_period)
    
    if len(history_df) > 0:
        # Create historical chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['bias_strength'],
            mode='lines+markers',
            name='Bias Strength',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='%{y:.1f}%<br>%{x}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add bullish/bearish zones
        fig.add_hrect(y0=15, y1=100, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=-100, y1=-15, fillcolor="red", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title=f"Bias Strength Over Last {history_period} Days",
            xaxis_title="Date",
            yaxis_title="Bias Strength (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Days Analyzed", len(history_df))
        
        with col2:
            bullish_days = len(history_df[history_df['overall_bias'] == 'Bullish'])
            st.metric("Bullish Days", bullish_days, delta=f"{bullish_days/len(history_df)*100:.1f}%")
        
        with col3:
            bearish_days = len(history_df[history_df['overall_bias'] == 'Bearish'])
            st.metric("Bearish Days", bearish_days, delta=f"{bearish_days/len(history_df)*100:.1f}%")
        
        with col4:
            avg_confidence = history_df['bias_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Show recent readings table
        with st.expander("📋 View Data Table"):
            display_history = history_df[['timestamp', 'overall_bias', 'bias_strength', 'bias_confidence']].copy()
            display_history['timestamp'] = pd.to_datetime(display_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_history['bias_strength'] = display_history['bias_strength'].apply(lambda x: f"{x:+.1f}%")
            display_history['bias_confidence'] = display_history['bias_confidence'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_history, use_container_width=True, height=300)
        
        # Export historical data
        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Historical Data (CSV)",
            data=csv_history,
            file_name=f"bias_history_{history_period}d_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("📭 No historical data yet. Click 'Backfill Last 30 Days' in the sidebar to populate history.")

# ============================================================================
# AUTO REFRESH
# ============================================================================

if auto_refresh:
    time.sleep(300)  # 5 minutes
    st.rerun()

# Footer
st.markdown("---")
st.caption("🚀 NQ Futures Macro Bias Engine | Built for 5-min trading | Updated with 13 factors & 4 data sources")
