"""
OASIS Visualization Module

Creates visualizations for the OASIS (Open, Autonomous, Symbiotic, Intelligent, Sustainable)
organizational health assessment model.

Based on the integration of Ulanowicz's ecosystem theory with Fath et al. (2019)
"Measuring regenerative economics: 10 principles and measures undergirding
systemic economic health" (Global Transitions 1, 15-27).
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional


# Color scheme for OASIS dimensions
OASIS_COLORS = {
    'open': '#3498db',        # Blue - connectivity/openness
    'autonomous': '#9b59b6',  # Purple - learning/cycling
    'symbiotic': '#2ecc71',   # Green - cooperation/balance
    'intelligent': '#f39c12', # Orange - intelligence/roles
    'sustainable': '#1abc9c'  # Teal - sustainability/vitality
}

# Status colors
STATUS_COLORS = {
    'HEALTHY': '#27ae60',
    'WARNING': '#f39c12',
    'CRITICAL': '#e74c3c'
}


def create_oasis_radar_chart(oasis_scores: Dict[str, float],
                              show_thresholds: bool = True,
                              title: str = "OASIS Health Profile") -> go.Figure:
    """
    Create a radar chart displaying all 5 OASIS dimension scores.

    Args:
        oasis_scores: Dictionary with scores for each dimension (0-100)
        show_thresholds: Whether to show healthy/warning threshold lines
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Prepare data - add the first point again to close the polygon
    dimensions = ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']
    dimension_keys = ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']

    scores = [oasis_scores.get(key, 0) for key in dimension_keys]
    scores_closed = scores + [scores[0]]  # Close the polygon
    dimensions_closed = dimensions + [dimensions[0]]

    fig = go.Figure()

    # Add threshold zones if requested
    if show_thresholds:
        # Critical zone (0-30)
        fig.add_trace(go.Scatterpolar(
            r=[30] * 6,
            theta=dimensions_closed,
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.1)',
            line=dict(color='rgba(231, 76, 60, 0.3)', width=1, dash='dot'),
            name='Critical Zone',
            showlegend=True
        ))

        # Warning zone (30-50)
        fig.add_trace(go.Scatterpolar(
            r=[50] * 6,
            theta=dimensions_closed,
            fill='toself',
            fillcolor='rgba(243, 156, 18, 0.1)',
            line=dict(color='rgba(243, 156, 18, 0.3)', width=1, dash='dot'),
            name='Warning Zone',
            showlegend=True
        ))

        # Healthy zone (50-100)
        fig.add_trace(go.Scatterpolar(
            r=[100] * 6,
            theta=dimensions_closed,
            fill='toself',
            fillcolor='rgba(39, 174, 96, 0.1)',
            line=dict(color='rgba(39, 174, 96, 0.3)', width=1, dash='dot'),
            name='Healthy Zone',
            showlegend=True
        ))

    # Add actual scores
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=dimensions_closed,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='rgb(52, 152, 219)', width=3),
        name='Current Profile',
        text=[f'{s:.0f}' for s in scores_closed],
        textposition='top center',
        mode='lines+markers+text',
        marker=dict(size=10, color='rgb(52, 152, 219)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                tickvals=[0, 25, 50, 75, 100],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, weight='bold'),
                gridcolor='lightgray'
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            x=1.1,
            y=0.5,
            font=dict(size=10)
        ),
        title=dict(
            text=title,
            font=dict(size=16, weight='bold'),
            x=0.5
        ),
        height=500,
        margin=dict(l=80, r=150, t=80, b=80)
    )

    return fig


def create_dimension_gauge(score: float, dimension: str,
                           status: str = None) -> go.Figure:
    """
    Create a gauge chart for a single OASIS dimension.

    Args:
        score: Score value (0-100)
        dimension: Dimension name (e.g., 'open', 'sustainable')
        status: Optional status override ('HEALTHY', 'WARNING', 'CRITICAL')

    Returns:
        Plotly Figure object
    """
    # Determine color based on score or status
    if status:
        bar_color = STATUS_COLORS.get(status, '#3498db')
    elif score >= 60:
        bar_color = STATUS_COLORS['HEALTHY']
    elif score >= 40:
        bar_color = STATUS_COLORS['WARNING']
    else:
        bar_color = STATUS_COLORS['CRITICAL']

    # Dimension display name
    dim_names = {
        'open': 'OPEN',
        'autonomous': 'AUTONOMOUS',
        'symbiotic': 'SYMBIOTIC',
        'intelligent': 'INTELLIGENT',
        'sustainable': 'SUSTAINABLE'
    }
    display_name = dim_names.get(dimension.lower(), dimension.upper())

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(size=24)),
        title=dict(text=display_name, font=dict(size=14, weight='bold')),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="darkgray",
                tickvals=[0, 25, 50, 75, 100]
            ),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 30], color="rgba(231, 76, 60, 0.2)"),
                dict(range=[30, 50], color="rgba(243, 156, 18, 0.2)"),
                dict(range=[50, 75], color="rgba(39, 174, 96, 0.2)"),
                dict(range=[75, 100], color="rgba(39, 174, 96, 0.3)")
            ],
            threshold=dict(
                line=dict(color="red", width=2),
                thickness=0.75,
                value=score
            )
        )
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white'
    )

    return fig


def create_all_dimension_gauges(oasis_profile: Dict[str, Any]) -> go.Figure:
    """
    Create a subplot with gauges for all 5 OASIS dimensions.

    Args:
        oasis_profile: Complete OASIS profile from OASISCalculator

    Returns:
        Plotly Figure with 5 gauge subplots
    """
    scores = oasis_profile['dimension_scores']
    status = oasis_profile['dimension_status']

    dimensions = ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']
    dim_names = ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']

    fig = make_subplots(
        rows=1, cols=5,
        specs=[[{'type': 'indicator'}] * 5],
        subplot_titles=dim_names
    )

    for i, (dim, name) in enumerate(zip(dimensions, dim_names)):
        score = scores[dim]
        dim_status = status[dim]

        # Determine color
        if dim_status == 'HEALTHY':
            bar_color = STATUS_COLORS['HEALTHY']
        elif dim_status == 'WARNING':
            bar_color = STATUS_COLORS['WARNING']
        else:
            bar_color = STATUS_COLORS['CRITICAL']

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number=dict(suffix="", font=dict(size=18)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickvals=[0, 50, 100]),
                    bar=dict(color=bar_color, thickness=0.7),
                    steps=[
                        dict(range=[0, 30], color="rgba(231, 76, 60, 0.15)"),
                        dict(range=[30, 50], color="rgba(243, 156, 18, 0.15)"),
                        dict(range=[50, 100], color="rgba(39, 174, 96, 0.15)")
                    ]
                )
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white'
    )

    return fig


def create_contribution_chart(dimension: str, metrics: Dict[str, float],
                               weights: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart showing metric contributions to a dimension score.

    Args:
        dimension: Dimension name
        metrics: Dictionary of normalized metric values
        weights: Dictionary of metric weights

    Returns:
        Plotly Figure object
    """
    # Filter to relevant metrics (those with weights)
    metric_names = list(weights.keys())
    contributions = []
    labels = []

    for metric in metric_names:
        weight = weights[metric]
        # Find the corresponding normalized metric value
        # Try different naming conventions
        value = metrics.get(f'norm_{metric}', metrics.get(metric, 0))
        contribution = value * weight * 100
        contributions.append(contribution)
        labels.append(metric.replace('_', ' ').title())

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=contributions,
        y=labels,
        orientation='h',
        marker_color=OASIS_COLORS.get(dimension.lower(), '#3498db'),
        text=[f'{c:.1f}' for c in contributions],
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(
            text=f'{dimension.upper()} Score Breakdown',
            font=dict(size=14)
        ),
        xaxis_title='Contribution to Score',
        yaxis_title='',
        height=250,
        margin=dict(l=120, r=20, t=50, b=40),
        xaxis=dict(range=[0, max(contributions) * 1.2] if contributions else [0, 100])
    )

    return fig


def create_overall_score_indicator(overall_score: float,
                                    overall_status: str) -> go.Figure:
    """
    Create a large indicator for the overall OASIS score.

    Args:
        overall_score: Overall weighted score (0-100)
        overall_status: Overall status ('HEALTHY', 'WARNING', 'CRITICAL')

    Returns:
        Plotly Figure object
    """
    color = STATUS_COLORS.get(overall_status, '#3498db')

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        number=dict(
            suffix="/100",
            font=dict(size=40, weight='bold')
        ),
        title=dict(
            text=f"Overall OASIS Health: {overall_status}",
            font=dict(size=18, weight='bold')
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=2,
                tickcolor="darkgray",
                tickvals=[0, 20, 40, 60, 80, 100]
            ),
            bar=dict(color=color, thickness=0.8),
            bgcolor="white",
            borderwidth=3,
            bordercolor="gray",
            steps=[
                dict(range=[0, 40], color="rgba(231, 76, 60, 0.25)"),
                dict(range=[40, 60], color="rgba(243, 156, 18, 0.25)"),
                dict(range=[60, 100], color="rgba(39, 174, 96, 0.25)")
            ]
        )
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='white'
    )

    return fig


def create_dimension_comparison_bar(oasis_scores: Dict[str, float]) -> go.Figure:
    """
    Create a horizontal bar chart comparing all dimension scores.

    Args:
        oasis_scores: Dictionary with scores for each dimension

    Returns:
        Plotly Figure object
    """
    dimensions = ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']
    dimension_keys = ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']

    scores = [oasis_scores.get(key, 0) for key in dimension_keys]
    colors = [OASIS_COLORS[key] for key in dimension_keys]

    fig = go.Figure(go.Bar(
        x=scores,
        y=dimensions,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.0f}' for s in scores],
        textposition='outside'
    ))

    # Add threshold lines
    fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Critical", annotation_position="top")
    fig.add_vline(x=50, line_dash="dash", line_color="orange", opacity=0.5,
                  annotation_text="Warning", annotation_position="top")

    fig.update_layout(
        title=dict(
            text="OASIS Dimension Scores",
            font=dict(size=16, weight='bold')
        ),
        xaxis=dict(
            title="Score",
            range=[0, 105],
            tickvals=[0, 25, 50, 75, 100]
        ),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=120, r=50, t=60, b=40)
    )

    return fig


def create_sustainability_detail_chart(sustainable_metrics: Dict[str, Any]) -> go.Figure:
    """
    Create a detailed visualization for the SUSTAINABLE dimension,
    showing the Window of Vitality position.

    Args:
        sustainable_metrics: Metrics dictionary from SUSTAINABLE dimension calculation

    Returns:
        Plotly Figure object
    """
    alpha = sustainable_metrics.get('relative_ascendency', 0.5)
    robustness = sustainable_metrics.get('robustness', 0)
    is_viable = sustainable_metrics.get('is_viable', False)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Window of Vitality Position", "Robustness Function"],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Left plot: Position in Window of Vitality
    # Add window zone
    fig.add_shape(
        type="rect",
        x0=0.2, x1=0.6, y0=0, y1=1,
        fillcolor="lightgreen",
        opacity=0.3,
        line=dict(color="green", width=2),
        row=1, col=1
    )

    # Add current position
    fig.add_trace(
        go.Scatter(
            x=[alpha],
            y=[0.5],
            mode='markers+text',
            marker=dict(
                size=20,
                color='green' if is_viable else 'red',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=[f'alpha={alpha:.3f}'],
            textposition='top center',
            name='Current Position',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add optimal point
    fig.add_trace(
        go.Scatter(
            x=[0.37],
            y=[0.5],
            mode='markers',
            marker=dict(size=12, color='gold', symbol='star'),
            name='Optimal (0.37)',
            showlegend=False
        ),
        row=1, col=1
    )

    # Right plot: Robustness curve
    alpha_range = np.linspace(0.01, 0.99, 100)
    robustness_values = -alpha_range * np.log(alpha_range)

    fig.add_trace(
        go.Scatter(
            x=alpha_range,
            y=robustness_values,
            mode='lines',
            line=dict(color='blue', width=2),
            name='R = -alpha*log(alpha)',
            showlegend=False
        ),
        row=1, col=2
    )

    # Mark current position on robustness curve
    fig.add_trace(
        go.Scatter(
            x=[alpha],
            y=[robustness],
            mode='markers',
            marker=dict(size=15, color='red', symbol='circle'),
            name=f'Current R={robustness:.3f}',
            showlegend=False
        ),
        row=1, col=2
    )

    # Mark optimal robustness
    optimal_r = -0.37 * np.log(0.37)
    fig.add_trace(
        go.Scatter(
            x=[0.37],
            y=[optimal_r],
            mode='markers',
            marker=dict(size=12, color='gold', symbol='star'),
            name='Optimal',
            showlegend=False
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Relative Ascendency (alpha)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, showticklabels=False, range=[0, 1], row=1, col=1)

    fig.update_xaxes(title_text="Relative Ascendency (alpha)", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Robustness (R)", row=1, col=2)

    fig.update_layout(
        height=350,
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=False
    )

    return fig


def create_recommendations_chart(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a visual representation of OASIS recommendations.

    Args:
        recommendations: List of recommendation dictionaries from OASISCalculator

    Returns:
        Plotly Figure object
    """
    if not recommendations:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No critical recommendations - system is healthy!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='green')
        )
        fig.update_layout(height=200)
        return fig

    # Prepare data
    priorities = [r['priority'] for r in recommendations]
    dimensions = [r['dimension'] for r in recommendations]
    issues = [r['issue'] for r in recommendations]

    # Priority colors
    priority_colors = {
        'CRITICAL': '#e74c3c',
        'HIGH': '#e67e22',
        'MEDIUM': '#f39c12',
        'LOW': '#3498db'
    }
    colors = [priority_colors.get(p, '#95a5a6') for p in priorities]

    # Create table-like visualization
    fig = go.Figure(go.Table(
        header=dict(
            values=['Priority', 'Dimension', 'Issue'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[priorities, dimensions, issues],
            fill_color=[
                [priority_colors.get(p, '#ecf0f1') for p in priorities],
                ['white'] * len(recommendations),
                ['white'] * len(recommendations)
            ],
            font=dict(color=['white' if p in ['CRITICAL', 'HIGH'] else 'black' for p in priorities] +
                           ['black'] * len(recommendations) * 2, size=11),
            align='left',
            height=30
        )
    ))

    fig.update_layout(
        title=dict(
            text="Recommended Actions",
            font=dict(size=14, weight='bold')
        ),
        height=50 + len(recommendations) * 35,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_oasis_summary_dashboard(oasis_profile: Dict[str, Any],
                                    recommendations: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a comprehensive OASIS dashboard with multiple visualizations.

    Args:
        oasis_profile: Complete OASIS profile from OASISCalculator
        recommendations: List of recommendations

    Returns:
        Plotly Figure with multiple subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'polar'}, {'type': 'indicator'}],
            [{'type': 'bar', 'colspan': 2}, None]
        ],
        subplot_titles=[
            'OASIS Health Profile',
            'Overall Score',
            'Dimension Scores'
        ],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.15
    )

    scores = oasis_profile['dimension_scores']
    overall = oasis_profile['overall_score']
    overall_status = oasis_profile['overall_status']

    # Radar chart data
    dimensions = ['OPEN', 'AUTO', 'SYMB', 'INTEL', 'SUST']
    dimension_keys = ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']
    radar_scores = [scores[k] for k in dimension_keys]
    radar_scores_closed = radar_scores + [radar_scores[0]]
    dimensions_closed = dimensions + [dimensions[0]]

    # Add radar chart
    fig.add_trace(
        go.Scatterpolar(
            r=radar_scores_closed,
            theta=dimensions_closed,
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.3)',
            line=dict(color='rgb(52, 152, 219)', width=2),
            name='Profile'
        ),
        row=1, col=1
    )

    # Add overall score gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=overall,
            number=dict(suffix="/100", font=dict(size=24)),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=STATUS_COLORS.get(overall_status, '#3498db')),
                steps=[
                    dict(range=[0, 40], color="rgba(231, 76, 60, 0.2)"),
                    dict(range=[40, 60], color="rgba(243, 156, 18, 0.2)"),
                    dict(range=[60, 100], color="rgba(39, 174, 96, 0.2)")
                ]
            )
        ),
        row=1, col=2
    )

    # Add bar chart
    full_names = ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']
    fig.add_trace(
        go.Bar(
            x=full_names,
            y=[scores[k] for k in dimension_keys],
            marker_color=[OASIS_COLORS[k] for k in dimension_keys],
            text=[f'{scores[k]:.0f}' for k in dimension_keys],
            textposition='outside'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50),
        polar=dict(
            radialaxis=dict(range=[0, 100], tickvals=[0, 50, 100])
        )
    )

    # Update bar chart y-axis
    fig.update_yaxes(range=[0, 105], row=2, col=1)

    return fig
