"""
Macro Bias Engine - Visualization Examples
==========================================

This script demonstrates how to create various visualizations
of the macro bias analysis results.

Requires: pip install matplotlib plotly
Run with: python visualizations.py
"""

import pandas as pd
import numpy as np
from macro_bias_engine import MacroBiasEngine, OutputFormatter

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


# ============================================================================
# MATPLOTLIB VISUALIZATIONS (Static Charts)
# ============================================================================

def create_factor_contribution_chart(results):
    """
    Create a horizontal bar chart showing factor contributions.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping.")
        return
    
    df = pd.DataFrame(results['factor_scores'])
    df = df.sort_values('Contribution_Pct', ascending=True)
    
    # Create color map based on direction
    colors = []
    for direction in df['Direction']:
        if direction == 'Bullish':
            colors.append('#10b981')  # Green
        elif direction == 'Bearish':
            colors.append('#ef4444')  # Red
        else:
            colors.append('#6b7280')  # Gray
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(df['Factor'], df['Contribution_Pct'], color=colors, alpha=0.8)
    
    # Customize
    ax.set_xlabel('Contribution to Overall Bias (%)', fontsize=12, fontweight='bold')
    ax.set_title('Factor Contributions to Macro Bias', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df['Contribution_Pct'])):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', 
                va='center', fontweight='bold', fontsize=10)
    
    # Add legend
    bullish_patch = mpatches.Patch(color='#10b981', label='Bullish', alpha=0.8)
    bearish_patch = mpatches.Patch(color='#ef4444', label='Bearish', alpha=0.8)
    neutral_patch = mpatches.Patch(color='#6b7280', label='Neutral', alpha=0.8)
    ax.legend(handles=[bullish_patch, bearish_patch, neutral_patch], 
              loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('factor_contributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: factor_contributions.png")
    plt.close()


def create_factor_scores_chart(results):
    """
    Create a scatter plot of factor scores vs weights.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping.")
        return
    
    df = pd.DataFrame(results['factor_scores'])
    
    # Create color map based on direction
    colors = []
    for direction in df['Direction']:
        if direction == 'Bullish':
            colors.append('#10b981')
        elif direction == 'Bearish':
            colors.append('#ef4444')
        else:
            colors.append('#6b7280')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with size based on contribution
    scatter = ax.scatter(df['Normalized_Score'], df['Weight'], 
                        s=df['Contribution_Pct']*20,  # Size based on contribution
                        c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add factor labels
    for idx, row in df.iterrows():
        ax.annotate(row['Factor'].replace('_', ' '), 
                   (row['Normalized_Score'], row['Weight']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add quadrant lines
    ax.axhline(y=df['Weight'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Customize
    ax.set_xlabel('Normalized Score (-1 to +1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Factor Weight', fontsize=12, fontweight='bold')
    ax.set_title('Factor Scores vs Weights\n(Bubble size = Contribution %)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add legend
    bullish_patch = mpatches.Patch(color='#10b981', label='Bullish', alpha=0.6)
    bearish_patch = mpatches.Patch(color='#ef4444', label='Bearish', alpha=0.6)
    ax.legend(handles=[bullish_patch, bearish_patch], 
              loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('factor_scores.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: factor_scores.png")
    plt.close()


def create_gauge_chart(results):
    """
    Create gauge charts for bias strength and confidence.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping.")
        return
    
    summary = results['summary']
    bias_strength = summary['bias_strength_pct']
    confidence = summary['bias_confidence_pct']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gauge 1: Bias Strength
    def create_gauge(ax, value, title, value_range=(-100, 100)):
        # Normalize value to 0-1 range
        normalized = (value - value_range[0]) / (value_range[1] - value_range[0])
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        ax.fill_between(theta[:33], 0, 1, color='#ef4444', alpha=0.3)  # Bearish
        ax.fill_between(theta[33:67], 0, 1, color='#fbbf24', alpha=0.3)  # Neutral
        ax.fill_between(theta[67:], 0, 1, color='#10b981', alpha=0.3)  # Bullish
        
        # Needle
        needle_angle = np.pi * normalized
        ax.plot([0, np.cos(needle_angle)], [0, np.sin(needle_angle)], 
               'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Styling
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 1.2)
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Labels
        ax.text(0, -0.3, f'{value:.1f}%', ha='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', fontsize=14, fontweight='bold')
        
        # Range labels
        ax.text(-1, 0, str(value_range[0]), ha='right', fontsize=10)
        ax.text(1, 0, str(value_range[1]), ha='left', fontsize=10)
    
    create_gauge(ax1, bias_strength, 'Bias Strength', (-100, 100))
    create_gauge(ax2, confidence, 'Confidence', (0, 100))
    
    plt.tight_layout()
    plt.savefig('gauge_charts.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: gauge_charts.png")
    plt.close()


# ============================================================================
# PLOTLY VISUALIZATIONS (Interactive Charts)
# ============================================================================

def create_interactive_dashboard(results):
    """
    Create an interactive dashboard with multiple charts.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping.")
        return
    
    df = pd.DataFrame(results['factor_scores'])
    summary = results['summary']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Factor Contributions', 'Factor Scores', 
                       'Bias Strength Gauge', 'Volatility Gauge'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Factor Contributions Bar Chart
    df_sorted = df.sort_values('Contribution_Pct', ascending=True)
    colors = ['#10b981' if d == 'Bullish' else '#ef4444' if d == 'Bearish' else '#6b7280' 
              for d in df_sorted['Direction']]
    
    fig.add_trace(
        go.Bar(
            y=df_sorted['Factor'],
            x=df_sorted['Contribution_Pct'],
            orientation='h',
            marker=dict(color=colors),
            text=df_sorted['Contribution_Pct'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Factor Scores Scatter Plot
    colors_scatter = ['#10b981' if d == 'Bullish' else '#ef4444' if d == 'Bearish' else '#6b7280' 
                     for d in df['Direction']]
    
    fig.add_trace(
        go.Scatter(
            x=df['Normalized_Score'],
            y=df['Weight'],
            mode='markers+text',
            marker=dict(
                size=df['Contribution_Pct']*2,
                color=colors_scatter,
                line=dict(color='black', width=1)
            ),
            text=df['Factor'].str.replace('_', ' '),
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>Score: %{x:.2f}<br>Weight: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Bias Strength Gauge
    bias_strength = summary['bias_strength_pct']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=bias_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bias Strength (%)"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [-100, -15], 'color': "#fee2e2"},
                    {'range': [-15, 15], 'color': "#fef3c7"},
                    {'range': [15, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': bias_strength
                }
            }
        ),
        row=2, col=1
    )
    
    # 4. Confidence Gauge
    confidence = summary['bias_confidence_pct']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981"},
                'steps': [
                    {'range': [0, 40], 'color': "#fee2e2"},
                    {'range': [40, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#d1fae5"}
                ]
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"<b>Macro Bias Dashboard - {summary['overall_bias']}</b>",
        title_font_size=20,
        showlegend=False,
        height=800,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Contribution (%)", row=1, col=1)
    fig.update_xaxes(title_text="Normalized Score", range=[-1.1, 1.1], row=1, col=2)
    fig.update_yaxes(title_text="Weight", row=1, col=2)
    
    # Save
    fig.write_html('interactive_dashboard.html')
    print("✅ Saved: interactive_dashboard.html (open in browser)")
    
    # Also show in browser (optional)
    # fig.show()


def create_factor_radar_chart(results):
    """
    Create a radar chart showing factor scores.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping.")
        return
    
    df = pd.DataFrame(results['factor_scores'])
    
    fig = go.Figure()
    
    # Normalize scores to 0-10 scale for radar chart
    normalized_values = ((df['Normalized_Score'] + 1) / 2 * 10).tolist()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=df['Factor'].str.replace('_', ' ').tolist(),
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgba(102, 126, 234, 1)', width=2),
        marker=dict(size=8, color='rgba(102, 126, 234, 1)'),
        hovertemplate='<b>%{theta}</b><br>Score: %{customdata:.2f}<extra></extra>',
        customdata=df['Normalized_Score']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[0, 5, 10],
                ticktext=['Bearish (-1)', 'Neutral (0)', 'Bullish (+1)']
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise"
            )
        ),
        title=f"<b>Factor Scores Radar Chart</b><br>" + 
              f"<i>Overall Bias: {results['summary']['overall_bias']}</i>",
        title_font_size=16,
        showlegend=False,
        height=600,
        template='plotly_white'
    )
    
    fig.write_html('factor_radar.html')
    print("✅ Saved: factor_radar.html (open in browser)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_all_visualizations():
    """
    Create all visualizations.
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS FOR MACRO BIAS ENGINE")
    print("="*70)
    
    # Run analysis
    print("\n📊 Running macro bias analysis...")
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    print("-" * 70)
    
    if MATPLOTLIB_AVAILABLE:
        print("\n[Matplotlib Visualizations]")
        create_factor_contribution_chart(results)
        create_factor_scores_chart(results)
        create_gauge_chart(results)
    else:
        print("\n⚠️  Matplotlib not available. Install with: pip install matplotlib")
    
    if PLOTLY_AVAILABLE:
        print("\n[Plotly Visualizations]")
        create_interactive_dashboard(results)
        create_factor_radar_chart(results)
    else:
        print("\n⚠️  Plotly not available. Install with: pip install plotly")
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    
    print("\n📁 Files created:")
    if MATPLOTLIB_AVAILABLE:
        print("   • factor_contributions.png")
        print("   • factor_scores.png")
        print("   • gauge_charts.png")
    if PLOTLY_AVAILABLE:
        print("   • interactive_dashboard.html (open in browser)")
        print("   • factor_radar.html (open in browser)")
    
    print("\n💡 Tip: Open the .html files in your web browser for interactive charts!")
    print("="*70 + "\n")


if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        print("\n❌ ERROR: No visualization libraries installed!")
        print("\nPlease install at least one:")
        print("  pip install matplotlib")
        print("  pip install plotly")
        print("\nOr install both:")
        print("  pip install matplotlib plotly")
    else:
        create_all_visualizations()
