"""
OASIS Report Generator Module

Generates publication-quality reports for the OASIS (Open, Autonomous, Symbiotic,
Intelligent, Sustainable) organizational health assessment.

Based on Fath et al. (2019) "Measuring regenerative economics: 10 principles
and measures undergirding systemic economic health" (Global Transitions 1, 15-27).
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class OASISReportGenerator:
    """
    Generate comprehensive OASIS assessment reports.

    Produces reports in multiple formats (text, HTML) with executive summaries,
    dimension analyses, and actionable recommendations.
    """

    # Dimension descriptions for reports
    DIMENSION_INFO = {
        'open': {
            'full_name': 'OPEN',
            'tagline': 'Ability to Interconnect and Exchange',
            'description': (
                'Measures the organization\'s capacity for cross-boundary communication, '
                'resource exchange, and information flow. Based on Fath et al. Principles '
                '1 (Cross-scale Circulation), 3 (Reliable Inputs), and 4 (Healthy Outputs).'
            ),
            'fath_principles': [1, 3, 4]
        },
        'autonomous': {
            'full_name': 'AUTONOMOUS',
            'tagline': 'Ability to Learn and Encode Routines',
            'description': (
                'Assesses the organization\'s capacity for learning, knowledge retention, '
                'and self-reinforcing feedback loops. Based on Fath et al. Principles '
                '2 (Regenerative Re-investment) and 9 (Constructive vs Extractive).'
            ),
            'fath_principles': [2, 9]
        },
        'symbiotic': {
            'full_name': 'SYMBIOTIC',
            'tagline': 'Human-Machine Integration and Balance',
            'description': (
                'Evaluates the balance and cooperation between organizational elements, '
                'including resource distribution and mutualistic relationships. Based on '
                'Fath et al. Principles 5 (Balance of Sizes) and 8 (Mutualism).'
            ),
            'fath_principles': [5, 8]
        },
        'intelligent': {
            'full_name': 'INTELLIGENT',
            'tagline': 'Leverage Diverse Intelligence Types',
            'description': (
                'Measures functional diversity and the ability to leverage different '
                'types of expertise. Based on Fath et al. Principles 7 (Sufficient Diversity) '
                'and 10 (Adaptive Learning).'
            ),
            'fath_principles': [7, 10]
        },
        'sustainable': {
            'full_name': 'SUSTAINABLE',
            'tagline': 'Balance Between Order and Freedom',
            'description': (
                'The central dimension assessing the organization\'s position in the '
                'Window of Vitality - the balance between efficiency and resilience. '
                'Based on Fath et al. Principle 6 (Resilience-Efficiency Balance).'
            ),
            'fath_principles': [6]
        }
    }

    def __init__(self, oasis_calculator, org_name: str = "Organization"):
        """
        Initialize report generator.

        Args:
            oasis_calculator: OASISCalculator instance with computed metrics
            org_name: Name of the organization being analyzed
        """
        self.oasis = oasis_calculator
        self.org_name = org_name
        self.timestamp = datetime.now()

        # Cache computed data
        self._profile = None
        self._interpretations = None
        self._recommendations = None

    def _get_profile(self) -> Dict[str, Any]:
        """Get cached or compute OASIS profile."""
        if self._profile is None:
            self._profile = self.oasis.get_oasis_profile()
        return self._profile

    def _get_interpretations(self) -> Dict[str, str]:
        """Get cached or compute interpretations."""
        if self._interpretations is None:
            self._interpretations = self.oasis.get_oasis_interpretation()
        return self._interpretations

    def _get_recommendations(self) -> List[Dict[str, Any]]:
        """Get cached or compute recommendations."""
        if self._recommendations is None:
            self._recommendations = self.oasis.get_recommendations()
        return self._recommendations

    def _status_symbol(self, status: str) -> str:
        """Get text symbol for status."""
        symbols = {
            'HEALTHY': '[OK]',
            'WARNING': '[!]',
            'CRITICAL': '[X]'
        }
        return symbols.get(status, '[?]')

    def generate_executive_summary(self) -> str:
        """
        Generate executive summary section.

        Returns:
            Formatted text summary
        """
        profile = self._get_profile()
        scores = profile['dimension_scores']
        overall = profile['overall_score']
        overall_status = profile['overall_status']
        dim_status = profile['dimension_status']

        # Count status categories
        healthy_count = sum(1 for s in dim_status.values() if s == 'HEALTHY')
        warning_count = sum(1 for s in dim_status.values() if s == 'WARNING')
        critical_count = sum(1 for s in dim_status.values() if s == 'CRITICAL')

        # Get key concerns
        concerns = []
        if dim_status['sustainable'] != 'HEALTHY':
            alpha = profile['dimension_details']['sustainable']['metrics']['relative_ascendency']
            if alpha < 0.2:
                concerns.append("system operating below viability threshold (too chaotic)")
            elif alpha > 0.6:
                concerns.append("system over-optimized and potentially brittle")

        if dim_status['autonomous'] == 'CRITICAL':
            concerns.append("weak feedback and learning mechanisms")

        if dim_status['open'] == 'CRITICAL':
            concerns.append("insufficient interconnectivity")

        summary = f"""
EXECUTIVE SUMMARY
=================

Organization: {self.org_name}
Assessment Date: {self.timestamp.strftime('%B %d, %Y')}

OVERALL OASIS HEALTH SCORE: {overall:.0f}/100 - {overall_status}

Dimension Assessment Summary:
- HEALTHY dimensions: {healthy_count}/5
- WARNING dimensions: {warning_count}/5
- CRITICAL dimensions: {critical_count}/5

Individual Scores:
- OPEN (Interconnectivity):     {scores['open']:.0f}/100 {self._status_symbol(dim_status['open'])}
- AUTONOMOUS (Learning):        {scores['autonomous']:.0f}/100 {self._status_symbol(dim_status['autonomous'])}
- SYMBIOTIC (Integration):      {scores['symbiotic']:.0f}/100 {self._status_symbol(dim_status['symbiotic'])}
- INTELLIGENT (Diversity):      {scores['intelligent']:.0f}/100 {self._status_symbol(dim_status['intelligent'])}
- SUSTAINABLE (Balance):        {scores['sustainable']:.0f}/100 {self._status_symbol(dim_status['sustainable'])}

"""
        if concerns:
            summary += "Key Concerns:\n"
            for concern in concerns:
                summary += f"  - {concern.capitalize()}\n"
        else:
            summary += "No critical concerns identified.\n"

        return summary

    def generate_dimension_section(self, dimension: str) -> str:
        """
        Generate detailed section for a specific dimension.

        Args:
            dimension: Dimension key (e.g., 'open', 'sustainable')

        Returns:
            Formatted text section
        """
        profile = self._get_profile()
        interpretations = self._get_interpretations()

        if dimension not in profile['dimension_details']:
            return f"Unknown dimension: {dimension}"

        details = profile['dimension_details'][dimension]
        info = self.DIMENSION_INFO[dimension]
        score = profile['dimension_scores'][dimension]
        status = profile['dimension_status'][dimension]

        section = f"""
{info['full_name']} DIMENSION
{'=' * (len(info['full_name']) + 10)}

Tagline: {info['tagline']}
Score: {score:.0f}/100 - Status: {status}

Description:
{info['description']}

Fath et al. (2019) Principles: {', '.join(f'P{p}' for p in info['fath_principles'])}

Interpretation:
{interpretations[dimension]}

Key Metrics:
"""
        # Add metrics
        metrics = details['metrics']
        for metric_name, value in metrics.items():
            if not metric_name.startswith('norm_') and not metric_name.endswith('_details'):
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        section += f"  - {metric_name.replace('_', ' ').title()}: {value:.3f}\n"
                    else:
                        section += f"  - {metric_name.replace('_', ' ').title()}: {value}\n"

        # Add weight contributions
        section += "\nScore Composition:\n"
        weights = details['weights']
        for metric, weight in weights.items():
            section += f"  - {metric.replace('_', ' ').title()}: {weight*100:.0f}% weight\n"

        return section

    def generate_recommendations(self) -> str:
        """
        Generate recommendations section.

        Returns:
            Formatted text recommendations
        """
        recommendations = self._get_recommendations()

        section = """
RECOMMENDATIONS
===============

"""
        if not recommendations:
            section += "No critical recommendations - the organization shows healthy patterns\nacross all OASIS dimensions.\n"
            return section

        section += f"Total recommendations: {len(recommendations)}\n\n"

        for i, rec in enumerate(recommendations, 1):
            section += f"""
Recommendation #{i}
Priority: {rec['priority']}
Dimension: {rec['dimension']}
Issue: {rec['issue']}
Action: {rec['action']}
Metrics to improve: {', '.join(rec.get('metrics_to_improve', ['N/A']))}

"""
        return section

    def generate_methodology_section(self) -> str:
        """
        Generate methodology section explaining the OASIS framework.

        Returns:
            Formatted methodology text
        """
        return """
METHODOLOGY
===========

The OASIS (Open, Autonomous, Symbiotic, Intelligent, Sustainable) assessment
framework integrates Ulanowicz's ecosystem theory with Fath et al.'s (2019)
10 Principles of Regenerative Economics.

Scientific Foundation:
- Ulanowicz, R.E. (2009) - Information theory metrics for ecosystem health
- Fath, B.D. et al. (2019) - 10 principles for regenerative economics
- Zorach, A.C. & Ulanowicz, R.E. (2003) - Network roles analysis

Scoring Methodology:
Each dimension is scored 0-100 based on weighted combinations of underlying
network metrics. Weights are calibrated based on theoretical importance and
empirical validation from ecological and organizational studies.

Status Thresholds (approximate):
- HEALTHY: Score >= 50-60 (varies by dimension)
- WARNING: Score >= 30-40
- CRITICAL: Score < 30-40

The overall OASIS score is a weighted average of all five dimensions,
with default equal weights (20% each) adjustable by the user.

Key Metrics by Dimension:
1. OPEN: connectance, flow diversity, clustering, betweenness centrality
2. AUTONOMOUS: Finn cycling index, reciprocity, AMI, autocatalytic index
3. SYMBIOTIC: Gini coefficient, modularity, effective nodes, mutualism
4. INTELLIGENT: number of roles, functional diversity, conditional entropy
5. SUSTAINABLE: relative ascendency (alpha), robustness, regenerative capacity

Window of Vitality (Sustainable Dimension):
The Window of Vitality (0.2 < alpha < 0.6) represents the range where
organizations balance efficiency with resilience. Peak robustness occurs
at alpha ~ 0.37, where the system maximizes its ability to maintain
function under perturbation.
"""

    def generate_references(self) -> str:
        """
        Generate scientific references section.

        Returns:
            Formatted references text
        """
        return """
REFERENCES
==========

Primary Reference:
Fath, B.D., Fiscus, D.A., Goerner, S.J., Berea, A., & Ulanowicz, R.E. (2019).
    Measuring regenerative economics: 10 principles and measures undergirding
    systemic economic health. Global Transitions, 1, 15-27.
    https://doi.org/10.1016/j.glt.2019.06.002

Supporting References:
Ulanowicz, R.E. (2009). A Third Window: Natural Life beyond Newton and Darwin.
    Templeton Foundation Press, West Conshohocken, PA.

Ulanowicz, R.E., Goerner, S.J., Lietaer, B., & Gomez, R. (2009). Quantifying
    sustainability: Resilience, efficiency and the return of information
    theory. Ecological Complexity, 6(1), 27-36.

Zorach, A.C., & Ulanowicz, R.E. (2003). Quantifying the complexity of flow
    networks: How many roles are there? Complexity, 8(3), 68-76.

Holling, C.S. (1973). Resilience and stability of ecological systems.
    Annual Review of Ecology and Systematics, 4(1), 1-23.
"""

    def generate_full_report(self, format: str = "text") -> str:
        """
        Generate complete OASIS assessment report.

        Args:
            format: Output format ("text" or "html")

        Returns:
            Complete formatted report
        """
        if format == "html":
            return self._generate_html_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        report = f"""
================================================================================
OASIS ORGANIZATIONAL HEALTH ASSESSMENT
{self.org_name}
================================================================================

Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Framework: OASIS (Open, Autonomous, Symbiotic, Intelligent, Sustainable)
Based on: Fath et al. (2019) - 10 Principles of Regenerative Economics

--------------------------------------------------------------------------------
"""
        report += self.generate_executive_summary()
        report += "\n" + "-" * 80 + "\n"

        # Add each dimension section
        for dim in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            report += self.generate_dimension_section(dim)
            report += "\n" + "-" * 80 + "\n"

        report += self.generate_recommendations()
        report += "\n" + "-" * 80 + "\n"

        report += self.generate_methodology_section()
        report += "\n" + "-" * 80 + "\n"

        report += self.generate_references()

        report += """
================================================================================
END OF OASIS ASSESSMENT REPORT
================================================================================
"""
        return report

    def _generate_html_report(self) -> str:
        """Generate HTML formatted report."""
        profile = self._get_profile()
        interpretations = self._get_interpretations()
        recommendations = self._get_recommendations()

        scores = profile['dimension_scores']
        overall = profile['overall_score']
        overall_status = profile['overall_status']
        dim_status = profile['dimension_status']

        # Status colors
        def status_color(status):
            return {
                'HEALTHY': '#27ae60',
                'WARNING': '#f39c12',
                'CRITICAL': '#e74c3c'
            }.get(status, '#888')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OASIS Assessment - {self.org_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f6fa;
            color: #2d3436;
        }}
        .header {{
            background: linear-gradient(135deg, #1abc9c, #3498db);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .overall-score {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .score-value {{
            font-size: 4em;
            font-weight: bold;
            color: {status_color(overall_status)};
        }}
        .dimension-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        .dimension-card h3 {{
            margin-top: 0;
        }}
        .metric {{
            display: inline-block;
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 5px;
        }}
        .recommendation {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid;
        }}
        .priority-CRITICAL {{ border-color: #e74c3c; }}
        .priority-HIGH {{ border-color: #e67e22; }}
        .priority-MEDIUM {{ border-color: #f39c12; }}
        .priority-LOW {{ border-color: #3498db; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OASIS Organizational Health Assessment</h1>
        <p><strong>{self.org_name}</strong></p>
        <p>Generated: {self.timestamp.strftime('%B %d, %Y at %H:%M')}</p>
    </div>

    <div class="overall-score">
        <div class="score-value">{overall:.0f}/100</div>
        <h2>Overall Status: {overall_status}</h2>
    </div>

    <h2>Dimension Scores</h2>
"""
        # Add dimension cards
        dim_info = self.DIMENSION_INFO
        dim_colors = {
            'open': '#3498db',
            'autonomous': '#9b59b6',
            'symbiotic': '#2ecc71',
            'intelligent': '#f39c12',
            'sustainable': '#1abc9c'
        }

        for dim in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            info = dim_info[dim]
            details = profile['dimension_details'][dim]

            html += f"""
    <div class="dimension-card" style="border-color: {dim_colors[dim]}">
        <h3>{info['full_name']} - {scores[dim]:.0f}/100
        <span style="color: {status_color(dim_status[dim])}">[{dim_status[dim]}]</span></h3>
        <p><em>{info['tagline']}</em></p>
        <p>{interpretations[dim]}</p>
        <div>
"""
            # Add key metrics
            metrics = details['metrics']
            for metric_name, value in list(metrics.items())[:4]:
                if isinstance(value, (int, float)) and not metric_name.startswith('norm_'):
                    if isinstance(value, float):
                        html += f'<span class="metric"><strong>{metric_name.replace("_", " ").title()}:</strong> {value:.3f}</span>\n'

            html += """
        </div>
    </div>
"""

        # Add recommendations
        html += "<h2>Recommendations</h2>\n"

        if not recommendations:
            html += '<p style="color: #27ae60;">No critical recommendations - all dimensions healthy!</p>\n'
        else:
            for rec in recommendations:
                html += f"""
    <div class="recommendation priority-{rec['priority']}">
        <strong>{rec['priority']}</strong> - {rec['dimension']}<br>
        <strong>Issue:</strong> {rec['issue']}<br>
        <strong>Action:</strong> {rec['action']}
    </div>
"""

        html += """
    <h2>References</h2>
    <p>Fath, B.D., Fiscus, D.A., Goerner, S.J., Berea, A., & Ulanowicz, R.E. (2019).
    <em>Measuring regenerative economics: 10 principles and measures undergirding
    systemic economic health.</em> Global Transitions, 1, 15-27.</p>

</body>
</html>
"""
        return html

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get summary data as a dictionary for programmatic access.

        Returns:
            Dictionary with all summary data
        """
        profile = self._get_profile()
        interpretations = self._get_interpretations()
        recommendations = self._get_recommendations()

        return {
            'organization': self.org_name,
            'timestamp': self.timestamp.isoformat(),
            'overall_score': profile['overall_score'],
            'overall_status': profile['overall_status'],
            'dimension_scores': profile['dimension_scores'],
            'dimension_status': profile['dimension_status'],
            'interpretations': interpretations,
            'recommendations': recommendations,
            'weights': profile['weights']
        }
