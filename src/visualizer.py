"""
Visualization module for Ulanowicz sustainability analysis.

This module creates interactive charts showing:
- Sustainability curve with window of viability
- Organization position on the curve
- Key metrics and indicators
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from .ulanowicz_calculator import UlanowiczCalculator
except ImportError:
    from ulanowicz_calculator import UlanowiczCalculator


class SustainabilityVisualizer:
    """
    Creates visualizations for Ulanowicz sustainability analysis.
    """
    
    def __init__(self, calculator: UlanowiczCalculator):
        """
        Initialize visualizer with calculator instance.
        
        Args:
            calculator: UlanowiczCalculator instance with computed metrics
        """
        self.calculator = calculator
        self.metrics = calculator.get_sustainability_metrics()
        self.extended_metrics = calculator.get_extended_metrics()
        self.assessments = calculator.assess_regenerative_health()
    
    def plot_sustainability_curve_matplotlib(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create sustainability curve visualization using matplotlib.
        
        Args:
            figsize: Figure size tuple (width, height)
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Ascendency vs Development Capacity
        development_capacity = self.metrics['development_capacity']
        ascendency = self.metrics['ascendency']
        lower_bound = self.metrics['viability_lower_bound']
        upper_bound = self.metrics['viability_upper_bound']
        
        # Create theoretical curve points
        x_points = np.linspace(0, development_capacity * 1.2, 100)
        y_points = x_points  # Theoretical maximum ascendency line
        
        ax1.plot(x_points, y_points, 'k--', alpha=0.3, label='Theoretical Maximum')
        
        # Plot window of viability
        ax1.axhspan(lower_bound, upper_bound, alpha=0.2, color='green', 
                   label='Window of Viability')
        ax1.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7, 
                   label='Viability Bounds')
        ax1.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.7)
        
        # Plot current position
        ax1.plot(development_capacity, ascendency, 'ro', markersize=12, 
                label='Current Position')
        
        # Add annotations
        ax1.annotate(f'Organization\nA={ascendency:.2f}\nC={development_capacity:.2f}',
                    xy=(development_capacity, ascendency),
                    xytext=(development_capacity * 0.7, ascendency * 1.2),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='center')
        
        ax1.set_xlabel('Development Capacity (C)')
        ax1.set_ylabel('Ascendency (A)')
        ax1.set_title('Sustainability Curve - Window of Viability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Extended metrics breakdown
        metrics_names = ['TST', 'AMI', 'Ascendency', 'Overhead', 'Robustness', 'Flow Div.', 'Efficiency']
        metrics_values = [
            self.extended_metrics['total_system_throughput'],
            self.extended_metrics['average_mutual_information'],
            self.extended_metrics['ascendency'],
            self.extended_metrics['overhead'],
            self.extended_metrics['robustness'] * 10,  # Scale for visibility
            self.extended_metrics['flow_diversity'],
            self.extended_metrics['network_efficiency'] * 10  # Scale for visibility
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, 
                      color=['blue', 'orange', 'green', 'purple', 'brown'])
        ax2.set_title('Key Sustainability Metrics')
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax2.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_sustainability_curve_plotly(self) -> go.Figure:
        """
        Create interactive sustainability curve using Plotly with extended metrics.
        
        Returns:
            Plotly Figure object
        """
        # Create subplots with extended layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sustainability Curve', 'Core Metrics', 
                          'Extended Regenerative Metrics', 'Network Flow Matrix',
                          'Robustness Analysis', 'System Health Assessment'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        development_capacity = self.metrics['development_capacity']
        ascendency = self.metrics['ascendency']
        lower_bound = self.metrics['viability_lower_bound']
        upper_bound = self.metrics['viability_upper_bound']
        
        # Plot 1: Sustainability curve
        # Theoretical maximum line
        x_theory = np.linspace(0, development_capacity * 1.2, 100)
        y_theory = x_theory
        
        fig.add_trace(
            go.Scatter(x=x_theory, y=y_theory, mode='lines',
                      name='Theoretical Maximum',
                      line=dict(dash='dash', color='gray')),
            row=1, col=1
        )
        
        # Window of viability
        fig.add_shape(
            type="rect",
            x0=0, y0=lower_bound, x1=development_capacity * 1.2, y1=upper_bound,
            fillcolor="green", opacity=0.2,
            line=dict(width=0),
            row=1, col=1
        )
        
        # Viability bounds
        fig.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                     annotation_text="Lower Bound", row=1, col=1)
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                     annotation_text="Upper Bound", row=1, col=1)
        
        # Current position
        fig.add_trace(
            go.Scatter(x=[development_capacity], y=[ascendency],
                      mode='markers',
                      marker=dict(size=15, color='red'),
                      name='Organization Position',
                      text=f'A={ascendency:.2f}<br>C={development_capacity:.2f}',
                      textposition="top center"),
            row=1, col=1
        )
        
        # Plot 2: Core metrics bar chart
        core_metrics_names = ['TST', 'AMI', 'Ascendency', 'Overhead', 'Dev. Capacity']
        core_metrics_values = [
            self.extended_metrics['total_system_throughput'],
            self.extended_metrics['average_mutual_information'],
            self.extended_metrics['ascendency'],
            self.extended_metrics['overhead'],
            self.extended_metrics['development_capacity']
        ]
        
        fig.add_trace(
            go.Bar(x=core_metrics_names, y=core_metrics_values,
                   name='Core Metrics',
                   marker_color=['blue', 'orange', 'green', 'purple', 'brown']),
            row=1, col=2
        )
        
        # Plot 3: Extended regenerative metrics
        extended_names = ['Robustness', 'Flow Diversity', 'Efficiency', 'Redundancy', 'Regen. Capacity']
        extended_values = [
            self.extended_metrics['robustness'],
            self.extended_metrics['flow_diversity'],
            self.extended_metrics['network_efficiency'],
            self.extended_metrics['redundancy'],
            self.extended_metrics['regenerative_capacity']
        ]
        
        fig.add_trace(
            go.Bar(x=extended_names, y=extended_values,
                   name='Extended Metrics',
                   marker_color=['#dc2626', '#06b6d4', '#a855f7', '#f59e0b', '#84cc16']),
            row=2, col=1
        )
        
        # Plot 4: Flow matrix heatmap
        fig.add_trace(
            go.Heatmap(z=self.calculator.flow_matrix,
                      x=self.calculator.node_names,
                      y=self.calculator.node_names,
                      colorscale='Viridis',
                      name='Flow Matrix'),
            row=2, col=2
        )
        
        # Plot 5: Robustness indicator
        robustness_value = self.extended_metrics['robustness']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=robustness_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Robustness"},
                delta={'reference': 0.25, 'relative': True},
                gauge={
                    'axis': {'range': [None, 0.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 0.3], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.25}}),
            row=3, col=1
        )
        
        # Plot 6: System health assessment
        health_categories = list(self.assessments.keys())
        health_scores = [1 if 'HIGH' in assessment or 'GOOD' in assessment or 'OPTIMAL' in assessment 
                        else 0.5 if 'MODERATE' in assessment or 'VIABLE' in assessment
                        else 0.2 for assessment in self.assessments.values()]
        
        colors = ['#10b981' if score == 1 else '#f59e0b' if score == 0.5 else '#dc2626' for score in health_scores]
        
        fig.add_trace(
            go.Bar(x=health_categories, y=health_scores,
                   name='Health Assessment',
                   marker_color=colors,
                   text=[assessment.split(' - ')[0] for assessment in self.assessments.values()],
                   textposition='inside'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Extended Regenerative Economics Analysis Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Development Capacity (C)", row=1, col=1)
        fig.update_yaxes(title_text="Ascendency (A)", row=1, col=1)
        fig.update_xaxes(title_text="Core Metrics", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_xaxes(title_text="Extended Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Health Categories", row=3, col=2)
        fig.update_yaxes(title_text="Health Score", row=3, col=2)
        
        return fig
    
    def create_summary_report(self) -> Dict[str, any]:
        """
        Create a comprehensive summary report.
        
        Returns:
            Dictionary containing summary information
        """
        sustainability_status = self.calculator.assess_sustainability()
        
        return {
            'organization_name': 'Current Organization',
            'analysis_date': None,  # Could add timestamp
            'sustainability_status': sustainability_status,
            'metrics': self.metrics,
            'recommendations': self._generate_recommendations(),
            'regenerative_assessment': self.assessments,
            'extended_metrics_summary': self._summarize_extended_metrics(),
            'network_properties': {
                'nodes': self.calculator.n_nodes,
                'total_flows': np.count_nonzero(self.calculator.flow_matrix),
                'density': np.count_nonzero(self.calculator.flow_matrix) / (self.calculator.n_nodes ** 2)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on current position.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        ascendency = self.metrics['ascendency']
        lower_bound = self.metrics['viability_lower_bound']
        upper_bound = self.metrics['viability_upper_bound']
        
        if ascendency < lower_bound:
            recommendations.extend([
                "System is too chaotic - increase organization and structure",
                "Establish clearer communication channels",
                "Implement more formal processes and procedures",
                "Reduce redundant or conflicting pathways"
            ])
        elif ascendency > upper_bound:
            recommendations.extend([
                "System is too rigid - increase flexibility and redundancy",
                "Create alternative pathways and backup systems",
                "Encourage innovation and adaptability",
                "Reduce over-optimization and allow for variation"
            ])
        else:
            if ascendency < (lower_bound + upper_bound) / 2:
                recommendations.append("System is viable but could benefit from slightly more organization")
            else:
                recommendations.append("System is viable but could benefit from slightly more flexibility")
        
        # General recommendations
        overhead_ratio = self.metrics['overhead_ratio']
        if overhead_ratio < 0.3:
            recommendations.append("Consider increasing system redundancy for resilience")
        elif overhead_ratio > 0.7:
            recommendations.append("Consider reducing inefficiencies to improve performance")
        
        return recommendations
    
    def _summarize_extended_metrics(self) -> Dict[str, str]:
        """
        Summarize extended metrics with interpretations.
        
        Returns:
            Dictionary of metric summaries
        """
        summaries = {}
        
        # Robustness summary
        robustness = self.extended_metrics['robustness']
        if robustness > 0.25:
            summaries['robustness'] = f"High robustness ({robustness:.3f}) - System well-balanced"
        elif robustness > 0.1:
            summaries['robustness'] = f"Moderate robustness ({robustness:.3f}) - Room for improvement"
        else:
            summaries['robustness'] = f"Low robustness ({robustness:.3f}) - System vulnerable"
        
        # Flow diversity summary
        diversity = self.extended_metrics['flow_diversity']
        max_diversity = np.log(self.calculator.n_nodes ** 2)
        diversity_ratio = diversity / max_diversity if max_diversity > 0 else 0
        summaries['flow_diversity'] = f"Flow diversity: {diversity:.3f} ({diversity_ratio:.1%} of maximum)"
        
        # Efficiency summary
        efficiency = self.extended_metrics['network_efficiency']
        if 0.2 <= efficiency <= 0.6:
            summaries['efficiency'] = f"Optimal efficiency ({efficiency:.3f}) - Within sustainable range"
        elif efficiency < 0.2:
            summaries['efficiency'] = f"Low efficiency ({efficiency:.3f}) - Underutilized potential"
        else:
            summaries['efficiency'] = f"High efficiency ({efficiency:.3f}) - Risk of brittleness"
        
        # Regenerative capacity summary
        regen_cap = self.extended_metrics['regenerative_capacity']
        summaries['regenerative_capacity'] = f"Regenerative capacity: {regen_cap:.3f}"
        
        return summaries
    
    def create_robustness_curve(self) -> go.Figure:
        """
        Create a specialized robustness curve visualization.
        
        Returns:
            Plotly Figure showing robustness across efficiency spectrum
        """
        # Generate efficiency range
        efficiency_range = np.linspace(0.01, 0.99, 100)
        robustness_values = []
        
        # Calculate robustness for each efficiency value
        development_capacity = self.extended_metrics['development_capacity']
        
        for eff in efficiency_range:
            # R = (A/C) * (1 - A/C) * log(C)
            robustness = eff * (1 - eff) * np.log(development_capacity) if development_capacity > 0 else 0
            robustness_values.append(max(0, robustness))
        
        # Create the plot
        fig = go.Figure()
        
        # Add robustness curve
        fig.add_trace(go.Scatter(
            x=efficiency_range,
            y=robustness_values,
            mode='lines',
            name='Robustness Curve',
            line=dict(width=3)
        ))
        
        # Add current position
        current_efficiency = self.extended_metrics['network_efficiency']
        current_robustness = self.extended_metrics['robustness']
        
        fig.add_trace(go.Scatter(
            x=[current_efficiency],
            y=[current_robustness],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current Position'
        ))
        
        # Add optimal point
        optimal_efficiency = 0.37  # Theoretical optimum
        optimal_robustness = optimal_efficiency * (1 - optimal_efficiency) * np.log(development_capacity)
        
        fig.add_trace(go.Scatter(
            x=[optimal_efficiency],
            y=[optimal_robustness],
            mode='markers',
            marker=dict(size=12, color='green', symbol='star'),
            name='Optimal Point'
        ))
        
        # Update layout
        fig.update_layout(
            title='System Robustness vs Network Efficiency',
            xaxis_title='Network Efficiency (A/C)',
            yaxis_title='Robustness',
            showlegend=True
        )
        
        return fig
    
    def save_visualization(self, filename: str, format: str = 'html'):
        """
        Save visualization to file.
        
        Args:
            filename: Output filename
            format: Output format ('html', 'png', 'pdf')
        """
        if format == 'html':
            fig = self.plot_sustainability_curve_plotly()
            fig.write_html(filename)
        elif format == 'robustness':
            fig = self.create_robustness_curve()
            fig.write_html(filename.replace('.html', '_robustness.html'))
        else:
            fig = self.plot_sustainability_curve_matplotlib()
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
            plt.close(fig)