"""
Macro Bias Engine - Streamlit Dashboard
========================================

A beautiful, interactive web dashboard for real-time macro market analysis.

Run locally: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import time

# Import the macro bias engine
try:
    from macro_bias_engine import MacroBiasEngine, OutputFormatter
except ImportError:
    st.error("❌ Error: macro_bias_engine.py not found. Please ensure it's in the same directory.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Macro Bias Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
        border-bottom: 3px solid #1f77b4;
    }
    .bullish {
        color: #00c853;
        font-weight: bold;
    }
    .bearish {
        color: #ff1744;
        font-weight: bold;
    }
    .neutral {
        color: #ffa726;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=900)  # Cache for 15 minutes
def run_analysis(custom_weights=None):
    """Run the macro bias analysis with caching."""
    if custom_weights:
        engine = MacroBiasEngine(weights=custom_weights)
    else:
        engine = MacroBiasEngine()
    
    results = engine.run_analysis()
    return results

@st.cache_data
def get_default_weights():
    """Get default factor weights."""
    return {
        '10Y_Treasury_Yield': 2.0,
        'DXY_Dollar_Index': 2.0,
        'M2_Money_Supply': 2.0,
        'Credit_Spreads_BAA_AAA': 1.5,
        'VIX_Index': 1.5,
        'Economic_Surprises': 1.0
    }

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/logo.png", width=100)
    st.title("⚙️ Configuration")
    
    # Custom weights toggle
    use_custom_weights = st.checkbox("🎛️ Use Custom Weights", value=False)
    
    custom_weights = None
    if use_custom_weights:
        st.markdown("### 📊 Adjust Factor Weights")
        st.caption("Higher weights = more influence on final bias")
        
        custom_weights = {
            '10Y_Treasury_Yield': st.slider(
                "10-Year Treasury Yield", 
                0.0, 5.0, 2.0, 0.5,
                help="Rising yields = bearish (monetary tightening)"
            ),
            'DXY_Dollar_Index': st.slider(
                "US Dollar Index (DXY)", 
                0.0, 5.0, 2.0, 0.5,
                help="Rising dollar = bearish (tight financial conditions)"
            ),
            'M2_Money_Supply': st.slider(
                "M2 Money Supply", 
                0.0, 5.0, 2.0, 0.5,
                help="Rising M2 = bullish (liquidity expansion)"
            ),
            'Credit_Spreads_BAA_AAA': st.slider(
                "Credit Spreads (BAA-AAA)", 
                0.0, 5.0, 1.5, 0.5,
                help="Widening spreads = bearish (credit stress)"
            ),
            'VIX_Index': st.slider(
                "VIX Index", 
                0.0, 5.0, 1.5, 0.5,
                help="Rising VIX = bearish (fear/uncertainty)"
            ),
            'Economic_Surprises': st.slider(
                "Economic Surprises", 
                0.0, 5.0, 1.0, 0.5,
                help="Positive surprises = bullish"
            ),
        }
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("⏰ Auto-refresh (5 min)", value=False)
    
    st.markdown("---")
    
    # Information
    st.markdown("### ℹ️ About")
    st.info("""
    **Macro Bias Engine**
    
    Analyzes macroeconomic factors to determine market bias and regime.
    
    📊 Data Sources:
    - FRED (Federal Reserve)
    - Yahoo Finance
    
    ⚡ Updates: Real-time
    
    🔒 Privacy: No data stored
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Header
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🌍 Macro Bias Engine")
    st.caption("Real-Time Quantitative Market Analysis")
with col3:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# Run analysis
try:
    with st.spinner("🔄 Analyzing macro factors... This may take 30-60 seconds..."):
        results = run_analysis(custom_weights)
        summary = results['summary']
        
except Exception as e:
    st.error(f"❌ Error running analysis: {e}")
    st.info("💡 **Troubleshooting:**\n- Check your internet connection\n- Ensure required packages are installed\n- Try refreshing the page")
    st.stop()

# ============================================================================
# KEY METRICS ROW
# ============================================================================

metric_cols = st.columns(4)

# Overall Bias
with metric_cols[0]:
    bias = summary['overall_bias']
    bias_class = bias.lower()
    emoji = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
    
    st.metric(
        label="Overall Bias",
        value=f"{emoji} {bias}",
        help="Market directional bias based on macro factors"
    )

# Bias Strength
with metric_cols[1]:
    strength = summary['bias_strength_pct']
    strength_delta = f"{'+' if strength > 0 else ''}{strength:.1f}%"
    
    st.metric(
        label="Bias Strength",
        value=f"{abs(strength):.1f}%",
        delta=strength_delta,
        delta_color="normal" if strength > 0 else "inverse",
        help="Strength of the bias signal (-100% to +100%)"
    )

# Confidence
with metric_cols[2]:
    confidence = summary['bias_confidence_pct']
    
    st.metric(
        label="Signal Confidence",
        value=f"{confidence:.1f}%",
        help="How confident we are in the bias signal (0-100%)"
    )

# Volatility & Regime
with metric_cols[3]:
    volatility = summary['volatility_pct']
    regime = summary['regime']
    
    st.metric(
        label=regime,
        value=f"{volatility:.1f}%",
        help="Current market volatility level"
    )

st.markdown("---")

# ============================================================================
# MAIN CONTENT - 2 COLUMNS
# ============================================================================

col_left, col_right = st.columns([2, 1])

# LEFT COLUMN - CHARTS
with col_left:
    
    # Factor Contributions Chart
    st.subheader("📊 Factor Contributions")
    
    df = pd.DataFrame(results['factor_scores'])
    df = df.sort_values('Contribution_Pct', ascending=True)
    
    # Create color mapping
    colors = []
    for direction in df['Direction']:
        if direction == 'Bullish':
            colors.append('#00c853')  # Green
        elif direction == 'Bearish':
            colors.append('#ff1744')  # Red
        else:
            colors.append('#ffa726')  # Orange
    
    fig_contributions = go.Figure()
    
    fig_contributions.add_trace(go.Bar(
        y=df['Factor'].str.replace('_', ' '),
        x=df['Contribution_Pct'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=df['Contribution_Pct'].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.1f}%<br><extra></extra>'
    ))
    
    fig_contributions.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Contribution to Overall Bias (%)",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    
    st.plotly_chart(fig_contributions, use_container_width=True)
    
    # Factor Scores vs Weights Scatter
    st.subheader("🎯 Factor Scores vs Weights")
    st.caption("Bubble size = contribution to bias")
    
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=df['Normalized_Score'],
        y=df['Weight'],
        mode='markers+text',
        marker=dict(
            size=df['Contribution_Pct']*2,
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            sizemode='diameter'
        ),
        text=df['Factor'].str.replace('_', ' '),
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>Score: %{x:.2f}<br>Weight: %{y:.1f}<br><extra></extra>'
    ))
    
    # Add reference lines
    fig_scatter.add_hline(y=df['Weight'].mean(), line_dash="dash", 
                         line_color="gray", opacity=0.5,
                         annotation_text="Avg Weight", annotation_position="right")
    fig_scatter.add_vline(x=0, line_color="black", line_width=1)
    
    fig_scatter.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Normalized Score (-1 to +1)",
        yaxis_title="Factor Weight",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', range=[-1.1, 1.1]),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# RIGHT COLUMN - GAUGES & INFO
with col_right:
    
    # Bias Strength Gauge
    st.subheader("📈 Bias Strength")
    
    strength_value = summary['bias_strength_pct']
    
    fig_gauge1 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=strength_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4", 'thickness': 0.75},
            'steps': [
                {'range': [-100, -15], 'color': "#ffcdd2"},
                {'range': [-15, 15], 'color': "#fff9c4"},
                {'range': [15, 100], 'color': "#c8e6c9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': strength_value
            }
        }
    ))
    
    fig_gauge1.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    st.plotly_chart(fig_gauge1, use_container_width=True)
    
    # Confidence Gauge
    st.subheader("🎯 Signal Confidence")
    
    confidence_value = summary['bias_confidence_pct']
    
    fig_gauge2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#00c853", 'thickness': 0.75},
            'steps': [
                {'range': [0, 40], 'color': "#ffcdd2"},
                {'range': [40, 70], 'color': "#fff9c4"},
                {'range': [70, 100], 'color': "#c8e6c9"}
            ]
        }
    ))
    
    fig_gauge2.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    st.plotly_chart(fig_gauge2, use_container_width=True)
    
    # Trading Signal Box
    st.subheader("💡 Trading Signal")
    
    # Generate signal
    if strength_value > 30 and confidence_value > 60:
        signal = "🟢 STRONG BUY"
        signal_color = "green"
    elif strength_value > 15 and confidence_value > 40:
        signal = "🟢 BUY"
        signal_color = "green"
    elif strength_value < -30 and confidence_value > 60:
        signal = "🔴 STRONG SELL"
        signal_color = "red"
    elif strength_value < -15 and confidence_value > 40:
        signal = "🔴 SELL"
        signal_color = "red"
    else:
        signal = "⚪ NEUTRAL / HOLD"
        signal_color = "orange"
    
    st.markdown(f"### :{signal_color}[{signal}]")
    
    # Risk level based on volatility
    if volatility < 33:
        risk_level = "🟢 Low Risk"
    elif volatility < 66:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"
    
    st.info(f"""
    **Market Regime:** {regime}  
    **Risk Level:** {risk_level}  
    **Volatility:** {volatility:.1f}%
    """)

st.markdown("---")

# ============================================================================
# DETAILED FACTOR TABLE
# ============================================================================

st.subheader("📋 Detailed Factor Analysis")

# Prepare table
df_table = pd.DataFrame(results['factor_scores'])
df_table = df_table.sort_values('Contribution_Pct', ascending=False)

# Format columns
df_display = df_table[['Factor', 'Direction', 'Normalized_Score', 'Weight', 
                       'Weighted_Score', 'Contribution_Pct', 'Current_Value']].copy()

df_display['Factor'] = df_display['Factor'].str.replace('_', ' ')
df_display['Normalized_Score'] = df_display['Normalized_Score'].round(3)
df_display['Weighted_Score'] = df_display['Weighted_Score'].round(3)
df_display['Contribution_Pct'] = df_display['Contribution_Pct'].round(2)
df_display['Current_Value'] = df_display['Current_Value'].round(2)

# Rename columns
df_display.columns = ['Factor', 'Direction', 'Score', 'Weight', 
                     'Weighted Score', 'Contribution %', 'Current Value']

# Display table with color coding
def color_direction(val):
    if val == 'Bullish':
        return 'background-color: #c8e6c9'
    elif val == 'Bearish':
        return 'background-color: #ffcdd2'
    else:
        return 'background-color: #fff9c4'

styled_table = df_display.style.applymap(color_direction, subset=['Direction'])

st.dataframe(styled_table, use_container_width=True, height=300)

# ============================================================================
# INTERPRETATION & INSIGHTS
# ============================================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Key Insights")
    
    # Find dominant factor
    dominant_factor = df_table.iloc[0]
    
    insights = f"""
    **Dominant Factor:** {dominant_factor['Factor'].replace('_', ' ')}  
    - Contributing {dominant_factor['Contribution_Pct']:.1f}% to overall bias
    - Current signal: {dominant_factor['Direction']}
    
    **Overall Assessment:**  
    - Market bias is **{summary['overall_bias']}** with **{confidence:.0f}%** confidence
    - Bias strength: **{abs(strength_value):.1f}%**
    - Current regime: **{regime}**
    """
    
    st.markdown(insights)

with col2:
    st.subheader("⚠️ Risk Considerations")
    
    risk_notes = []
    
    if confidence < 50:
        risk_notes.append("⚠️ **Low confidence signal** - Consider waiting for stronger confirmation")
    
    if volatility > 66:
        risk_notes.append("⚠️ **High volatility regime** - Reduce position sizes")
    
    # Check for conflicting signals
    bullish_count = len(df_table[df_table['Direction'] == 'Bullish'])
    bearish_count = len(df_table[df_table['Direction'] == 'Bearish'])
    
    if abs(bullish_count - bearish_count) <= 1:
        risk_notes.append("⚠️ **Mixed signals** - Factors are conflicting")
    
    if dominant_factor['Contribution_Pct'] > 40:
        risk_notes.append(f"⚠️ **Single factor dominance** - {dominant_factor['Factor'].replace('_', ' ')} driving {dominant_factor['Contribution_Pct']:.0f}% of signal")
    
    if not risk_notes:
        st.success("✅ No major risk warnings at this time")
    else:
        for note in risk_notes:
            st.warning(note)

# ============================================================================
# EXPORT OPTIONS
# ============================================================================

st.markdown("---")

st.subheader("💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    # JSON export
    import json
    json_data = OutputFormatter.to_json(results)
    st.download_button(
        label="📄 Download JSON",
        data=json_data,
        file_name=f"macro_bias_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

with col2:
    # CSV export
    csv_data = df_table.to_csv(index=False)
    st.download_button(
        label="📊 Download CSV",
        data=csv_data,
        file_name=f"factor_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

with col3:
    # Summary report
    summary_text = f"""MACRO BIAS ENGINE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL ASSESSMENT
------------------
Bias: {summary['overall_bias']}
Strength: {strength_value:+.2f}%
Confidence: {confidence:.2f}%
Volatility: {volatility:.2f}%
Regime: {regime}

TRADING SIGNAL: {signal}

TOP FACTORS
-----------
{df_table.head(3)[['Factor', 'Direction', 'Contribution_Pct']].to_string(index=False)}
"""
    
    st.download_button(
        label="📋 Download Report",
        data=summary_text,
        file_name=f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# ============================================================================
# FOOTER & DISCLAIMER
# ============================================================================

st.markdown("---")

st.caption("""
**⚠️ DISCLAIMER:** This tool is for informational and educational purposes only. 
It does NOT constitute financial advice. Always do your own research and consult 
with licensed financial professionals before making investment decisions. 
Past performance does not guarantee future results.
""")

st.caption("Built with ❤️ using Python, Streamlit, and real-time macro data | MIT License")

# ============================================================================
# AUTO-REFRESH
# ============================================================================

if auto_refresh:
    time.sleep(300)  # 5 minutes
    st.rerun()
