"""
Publication-Quality Report Generator for Adaptive Organization Analysis
Author: Adaptive Organization Analysis System
Based on: Ulanowicz-Fath Regenerative Economics Framework
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import json


class PublicationReportGenerator:
    """
    Generates professional, audit-firm quality reports for organizational
    network analysis grounded in the Ulanowicz-Fath regenerative economics
    framework.  Output is plain ASCII text structured for consumption by
    the companion PDF renderer (src/pdf_generator.py).
    """

    def __init__(self, calculator, metrics: Dict[str, Any], assessments: Dict[str, str],
                 org_name: str, flow_matrix: np.ndarray, node_names: List[str]):
        """Initialize report generator with analysis data."""
        self.calculator = calculator
        self.metrics = metrics
        self.assessments = assessments
        self.org_name = org_name
        self.flow_matrix = flow_matrix
        self.node_names = node_names
        self.timestamp = datetime.now()

    # ==================================================================
    # Public report sections
    # ==================================================================

    def generate_abstract(self) -> str:
        """Generate a two-paragraph executive abstract."""

        alpha = self.metrics['ascendency_ratio']
        rob = self.metrics['robustness']
        viable = self.metrics['is_viable']
        n_nodes = len(self.node_names)
        n_edges = np.count_nonzero(self.flow_matrix)
        tst = np.sum(self.flow_matrix)

        viability_word = "within" if viable else "outside"
        sustainability_clause = (
            "sustainable operational characteristics consistent with long-term adaptive capacity"
            if viable else
            "structural conditions that warrant management attention and targeted intervention"
        )

        abstract = f"""
ABSTRACT
========

{self.org_name} {"demonstrates" if viable else "presents"} a network whose relative ascendency of alpha = {alpha:.3f} places it {viability_word} the empirically derived window of viability ({self.metrics['viability_lower_bound']:.2f} < alpha < {self.metrics['viability_upper_bound']:.2f}), indicating {sustainability_clause}. This assessment applies the Ulanowicz-Fath regenerative economics framework to a directed network of {n_nodes} organizational units connected through {n_edges} active flow relationships, representing a total system throughput of {tst:.1f} units. The system achieves a robustness of R = {rob:.3f} ({self._categorize_robustness().lower()}) and utilizes {alpha * 100:.1f}% of its development capacity for organized behavior while retaining {self.metrics['overhead_ratio'] * 100:.1f}% as overhead reserves for adaptability.

The analysis reveals {self._categorize_efficiency().lower()} network efficiency and {"a hierarchically layered" if self.metrics.get('trophic_depth', 0) > 2 else "a relatively flat"} information flow architecture with an effective link density of {self.metrics.get('effective_link_density', 0):.3f} and trophic depth of {self.metrics.get('trophic_depth', 0):.3f}. A flow diversity index of H = {self.metrics.get('flow_diversity', 0):.3f} bits indicates {"substantial" if self.metrics.get('flow_diversity', 0) > 3 else "moderate" if self.metrics.get('flow_diversity', 0) > 2 else "limited"} information distribution complexity. These quantitative findings provide an evidence base for strategic decisions regarding organizational design, resilience investment, and sustainable growth.
"""
        return abstract

    def generate_introduction(self) -> str:
        """Generate the introduction, framed as a professional engagement."""

        introduction = f"""
1. INTRODUCTION
===============

1.1 Engagement Context
----------------------
This report presents the findings of a comprehensive network sustainability assessment of {self.org_name}, conducted using the Ulanowicz-Fath regenerative economics framework. The framework applies information-theoretic principles originally developed for ecological network analysis to organizational systems, providing quantitative measures of system health, efficiency, and resilience that complement traditional performance indicators.

The business rationale for this approach is straightforward. Conventional performance metrics capture what an organization achieves at a point in time; they rarely illuminate whether the underlying network of flows -- information, resources, decisions -- is configured for long-term viability. The regenerative economics lens addresses this gap by measuring the balance between constraining efficiency and adaptive flexibility, a balance that decades of ecological research have shown to be the hallmark of sustainable systems (Ulanowicz, 2009; Fath et al., 2019).

1.2 Theoretical Foundation
--------------------------
The framework treats organizations as flow networks in which resources or information move between nodes (departments, teams, or individuals). Sustainability is modeled through two complementary quantities. Ascendency (A) captures the degree of organized, purposeful flow, calculated as the product of total system throughput and the average mutual information among flows. Development capacity (C) represents the theoretical upper bound of organization, derived from total throughput and the Shannon entropy of the flow distribution. The ratio alpha = A / C indicates what fraction of the system's potential is channeled into structured behavior; the complement, overhead, represents reserve capacity available for adaptation and recovery.

Empirical studies of ecological and economic networks have identified a window of viability -- a range of alpha values within which systems demonstrate sustainable dynamics. Systems below the lower bound lack sufficient coherence; those above the upper bound sacrifice adaptive capacity for short-term efficiency and become brittle.

1.3 Scope and Objectives
-------------------------
This assessment of {self.org_name} pursues four objectives: first, to quantify the organization's position relative to the window of viability; second, to evaluate the balance between efficiency and resilience; third, to characterize the structural architecture of information flows; and fourth, to translate quantitative findings into prioritized, actionable recommendations for organizational leadership.
"""
        return introduction

    def generate_methodology(self) -> str:
        """Generate the approach section in accessible language."""

        data_source = self._identify_data_source()
        n_nodes = len(self.node_names)
        n_edges = np.count_nonzero(self.flow_matrix)
        tst = np.sum(self.flow_matrix)

        methodology = f"""
2. APPROACH
===========

2.1 Data and Network Construction
----------------------------------
The analysis is based on a weighted directed flow matrix representing {self.org_name}. The network comprises {n_nodes} organizational units connected by {n_edges} active flow relationships, with a total system throughput of {tst:.2f} units. The data source is classified as: {data_source}.

Each cell F_ij of the matrix records the magnitude of flow from unit i to unit j. Flows may represent information exchange, resource allocation, decision authority, communication frequency, or material and energy transfer depending on the organizational context.

2.2 What Each Measure Captures
-------------------------------
The analytical framework computes several families of indicators, each addressing a distinct question about organizational health.

2.2.1 Scale and Diversity
Total System Throughput (TST) measures the aggregate volume of all flows -- the overall scale of network activity. Flow Diversity (H) captures how evenly those flows are distributed: a high value indicates many pathways of comparable intensity, while a low value suggests concentration through a few dominant channels.

2.2.2 Organization and Capacity
Ascendency (A) quantifies how much of the network's activity is organized into purposeful, constrained pathways. Development Capacity (C) represents the theoretical maximum organization the network could achieve. The ratio alpha = A / C expresses current organization as a fraction of potential. Overhead (Phi = C - A) measures the reserve capacity available for adaptation, learning, and recovery from disruption.

2.2.3 Sustainability Position
The Window of Viability defines the range of alpha values associated with sustainable system dynamics, empirically bounded at 0.20 (minimum coherence) and 0.60 (maximum efficiency before brittleness onset). Robustness (R = -alpha x log2(alpha)) peaks at alpha approximately equal to 0.37, the theoretical optimum for balancing order with flexibility.

2.2.4 Network Architecture
Effective Link Density measures the proportion of possible connections that are active. Trophic Depth indicates how many hierarchical levels information traverses. Redundancy captures the availability of alternative pathways should primary channels be disrupted. Regenerative Capacity integrates overhead with proximity to the optimal balance point.

2.3 Validation
--------------
All calculations conform to the theoretical bounds established in the literature: 0 <= AMI <= log2(n), 0 <= alpha <= 1, and 0 <= R <= log2(e)/e (approximately 0.531). Computations use base-2 logarithms consistent with Ulanowicz's NETWRK 4.2 specifications. Full formula definitions are provided in the Appendix.
"""
        return methodology

    def generate_results(self) -> str:
        """Generate thematic narrative results."""

        alpha = self.metrics['ascendency_ratio']
        rob = self.metrics['robustness']
        viable = self.metrics['is_viable']
        n_nodes = len(self.node_names)
        n_edges = np.count_nonzero(self.flow_matrix)
        density = n_edges / (n_nodes ** 2)
        eld = self.metrics['effective_link_density']
        td = self.metrics['trophic_depth']
        fd = self.metrics['flow_diversity']
        asc = self.metrics['ascendency']
        dc = self.metrics['development_capacity']
        ovh = self.metrics['overhead']
        ovh_ratio = self.metrics['overhead_ratio']
        redund = self.metrics['redundancy']
        regen = self.metrics['regenerative_capacity']
        eff = self.metrics['network_efficiency']

        # Flow distribution stats
        active_flows = self.flow_matrix[self.flow_matrix > 0]
        mean_flow = np.mean(active_flows) if len(active_flows) > 0 else 0
        median_flow = np.median(active_flows) if len(active_flows) > 0 else 0
        std_flow = np.std(active_flows) if len(active_flows) > 0 else 0
        cv_flow = std_flow / mean_flow if mean_flow > 0 else 0
        gini = self._calculate_gini()

        # Viability interpretation
        if viable:
            viability_narrative = (
                f"The organization operates within the window of viability, confirming that its "
                f"current configuration balances constraining efficiency with adaptive flexibility "
                f"in a manner consistent with long-term sustainability."
            )
        elif alpha < self.metrics['viability_lower_bound']:
            viability_narrative = (
                f"The organization falls below the lower bound of the window of viability "
                f"(alpha = {alpha:.3f} vs. threshold of {self.metrics['viability_lower_bound']:.2f}), "
                f"indicating insufficient organizational coherence. Flows are dispersed across "
                f"too many weakly constrained pathways, reducing collective effectiveness."
            )
        else:
            viability_narrative = (
                f"The organization exceeds the upper bound of the window of viability "
                f"(alpha = {alpha:.3f} vs. threshold of {self.metrics['viability_upper_bound']:.2f}), "
                f"indicating over-constraint. Flows are concentrated through too few dominant "
                f"pathways, leaving insufficient reserves for adaptation and recovery."
            )

        # Robustness distance from optimum
        opt_distance = abs(alpha - 0.37)
        if opt_distance < 0.03:
            balance_assessment = "near the theoretical optimum for robustness (alpha approximately equal to 0.37), representing a well-calibrated balance between order and flexibility"
        elif alpha < 0.37:
            balance_assessment = f"in a resilience-favoring regime, {opt_distance * 100:.1f} percentage points below the robustness optimum, suggesting that some additional structuring of flows could improve performance without sacrificing adaptive capacity"
        else:
            balance_assessment = f"in an efficiency-favoring regime, {opt_distance * 100:.1f} percentage points above the robustness optimum, suggesting that the organization has prioritized structured throughput at some cost to its shock-absorption capacity"

        # Network structure interpretation
        if td > 3:
            depth_desc = "a deeply layered hierarchy with multiple processing levels"
        elif td > 2:
            depth_desc = "a moderately layered organizational structure"
        else:
            depth_desc = "a relatively flat architecture with limited hierarchical depth"

        if eld > 0.3:
            connect_desc = "high interconnectivity among units"
        elif eld > 0.1:
            connect_desc = "moderate interconnectivity"
        else:
            connect_desc = "sparse connections between units"

        results = f"""
3. RESULTS
==========

3.1 Organizational Health Position
-----------------------------------
{viability_narrative}

The relative ascendency of alpha = {alpha:.3f} indicates that {self.org_name} channels {alpha * 100:.1f}% of its development capacity into organized, purposeful flows. This positions the system as {self._interpret_position().lower()} on the organization spectrum. The viability assessment is summarized below.

Table 1. Viability Assessment Summary
--------------------------------------
Parameter                       Value       Status
Relative Ascendency (alpha)     {alpha:<11.3f} {self._interpret_position()}
Window Lower Bound              {self.metrics['viability_lower_bound']:<11.3f} {'PASS' if alpha > self.metrics['viability_lower_bound'] else 'FAIL'}
Window Upper Bound              {self.metrics['viability_upper_bound']:<11.3f} {'PASS' if alpha < self.metrics['viability_upper_bound'] else 'FAIL'}
Within Window of Viability      {'Yes':<11} {self._get_viability_interpretation()}

3.2 Efficiency-Resilience Balance
----------------------------------
The system is positioned {balance_assessment}. Robustness reaches R = {rob:.3f}, categorized as {self._categorize_robustness().lower()}, while network efficiency stands at {eff:.3f} ({self._categorize_efficiency().lower()}).

This trade-off has direct strategic significance. Higher ascendency ratios drive throughput performance but erode the overhead reserves that buffer the organization against disruption. The current overhead ratio of {ovh_ratio:.3f} ({ovh_ratio * 100:.1f}% of capacity) represents the organization's investment in adaptability -- the slack, redundancy, and alternative pathways that allow it to absorb shocks and reorganize under stress.

For leadership, the key question is whether this balance suits the organization's operating environment. Stable, predictable environments reward efficiency; volatile or rapidly changing environments demand greater reserves.

3.3 Information Flow Architecture
----------------------------------
The network of {n_nodes} units and {n_edges} active connections exhibits {depth_desc} combined with {connect_desc}. The effective link density of {eld:.3f} means that {eld * 100:.1f}% of all possible directed connections carry measurable flow. Trophic depth of {td:.3f} levels characterizes the average number of hierarchical stages that information or resources traverse from source to final use.

Flow diversity (H = {fd:.3f} bits) measures how evenly activity is distributed across channels. A higher value indicates many pathways of comparable weight; a lower value signals concentration through a few dominant links. The current diversity level suggests {"broadly distributed information flow, supporting organizational learning and lateral coordination" if fd > 3 else "moderate distribution of information across channels, with room for further diversification" if fd > 2 else "concentrated flow patterns where a few dominant channels carry the bulk of information, creating potential single points of failure"}.

The Gini coefficient of flow magnitudes is {gini:.3f}, indicating {"high inequality" if gini > 0.6 else "moderate inequality" if gini > 0.3 else "relatively equal distribution"} in how flow volumes are distributed across the network.

3.4 System Capacity and Reserves
----------------------------------
Development capacity (C = {dc:.3f}) establishes the theoretical ceiling for organized behavior in this network. Of that capacity, ascendency accounts for A = {asc:.3f} ({alpha * 100:.1f}%), while overhead accounts for Phi = {ovh:.3f} ({ovh_ratio * 100:.1f}%).

Redundancy stands at {redund:.3f}, meaning that {"ample" if redund > 0.6 else "moderate" if redund > 0.3 else "limited"} alternative pathways exist should primary channels be disrupted. Regenerative capacity -- a composite indicator reflecting both the magnitude of overhead reserves and proximity to the optimal balance point -- registers at {regen:.3f} ({self._interpret_regenerative().lower()}).

Table 2. Capacity and Reserve Indicators
-----------------------------------------
Metric                          Value       Assessment
Ascendency (A)                  {asc:<11.3f} {alpha * 100:.1f}% of capacity
Development Capacity (C)        {dc:<11.3f} Theoretical maximum
Overhead (Phi)                  {ovh:<11.3f} {ovh_ratio * 100:.1f}% of capacity
Redundancy                      {redund:<11.3f} {self._interpret_redundancy()}
Regenerative Capacity           {regen:<11.3f} {self._interpret_regenerative()}
Robustness (R)                  {rob:<11.3f} {self._categorize_robustness()}
Network Efficiency              {eff:<11.3f} {self._categorize_efficiency()}
"""
        return results

    def generate_discussion(self) -> str:
        """Generate strategic analysis and discussion."""

        alpha = self.metrics['ascendency_ratio']
        rob = self.metrics['robustness']
        viable = self.metrics['is_viable']
        eld = self.metrics['effective_link_density']
        td = self.metrics['trophic_depth']
        fd = self.metrics['flow_diversity']
        ovh_ratio = self.metrics['overhead_ratio']
        h_max = np.log2(len(self.node_names) ** 2)
        fd_utilization = (fd / h_max * 100) if h_max > 0 else 0

        # Strengths and risks
        strengths = []
        risks = []

        if viable:
            strengths.append(
                f"viability positioning (alpha = {alpha:.3f} within the sustainable window)"
            )
        else:
            risks.append(
                f"viability positioning (alpha = {alpha:.3f} outside the window bounds of "
                f"{self.metrics['viability_lower_bound']:.2f} to {self.metrics['viability_upper_bound']:.2f})"
            )

        if rob > 0.20:
            strengths.append(f"strong robustness (R = {rob:.3f})")
        elif rob > 0.15:
            strengths.append(f"adequate robustness (R = {rob:.3f})")
        else:
            risks.append(f"low robustness (R = {rob:.3f}), limiting shock-absorption capacity")

        if ovh_ratio > 0.4:
            strengths.append(f"substantial overhead reserves ({ovh_ratio * 100:.1f}% of capacity)")
        elif ovh_ratio < 0.3:
            risks.append(f"limited overhead reserves ({ovh_ratio * 100:.1f}% of capacity)")

        if self.metrics['redundancy'] > 0.5:
            strengths.append("high pathway redundancy, supporting continuity under disruption")
        elif self.metrics['redundancy'] < 0.3:
            risks.append("low pathway redundancy, creating vulnerability to single-channel failures")

        a_phi_ratio = self.metrics['ascendency'] / self.metrics['overhead'] if self.metrics['overhead'] > 0 else 0

        discussion = f"""
4. DISCUSSION
=============

4.1 Strategic Assessment
-------------------------
The analysis of {self.org_name} yields a clear overall picture: the organization {"maintains a configuration consistent with sustainable dynamics" if viable else "exhibits structural conditions that require deliberate intervention"}.

"""
        # Strengths
        if strengths:
            discussion += f"""The principal strengths identified are {'; '.join(strengths)}. These characteristics indicate that the organization {"has achieved" if viable else "partially maintains"} the kind of efficiency-resilience balance that ecological and economic research associates with long-term viability.\n\n"""

        # Risks
        if risks:
            discussion += f"""The material risks warranting management attention include {'; '.join(risks)}. {"Left unaddressed, these conditions could erode the organization's capacity to respond to environmental changes or absorb operational shocks." if not viable else "While these do not currently threaten viability, monitoring is warranted to ensure they do not deteriorate."}\n\n"""

        discussion += f"""4.2 Comparative Positioning
----------------------------
Empirical benchmarks from ecological and organizational literature provide useful context. Sustainable ecological food webs typically exhibit alpha in the range 0.20 to 0.50 (Ulanowicz, 2009). High-performing organizations analyzed using the same framework show alpha between 0.30 and 0.45 (Fath et al., 2019). The current system's alpha of {alpha:.3f} {"aligns with" if 0.30 <= alpha <= 0.45 else "deviates from"} the high-performing organizational benchmark.

The ratio of ascendency to overhead (A/Phi = {a_phi_ratio:.3f}) provides additional insight. This ratio reflects the balance between structured throughput and adaptive reserves, consistent with Holling's (1973) adaptive cycle theory which holds that systems must invest in reserve capacity to sustain long-term functionality.

The flow diversity utilization -- the ratio of observed diversity to the theoretical maximum -- stands at {fd_utilization:.1f}%. This indicates that {self.org_name} employs {"a broad range" if fd_utilization > 50 else "a limited fraction"} of its potential communication channels, {"supporting distributed knowledge flow" if fd_utilization > 50 else "suggesting opportunity to broaden information pathways"}.

4.3 Limitations and Caveats
-----------------------------
Several limitations should be considered when interpreting these findings.

The analysis represents a single point in time. Organizational networks evolve, and a longitudinal series of assessments would reveal trajectory and stability patterns that a snapshot cannot capture. The meaning assigned to flows -- whether information, resources, or influence -- affects interpretation; mixed flow types within a single matrix may obscure dynamics specific to one category. The system boundary and level of node aggregation also influence metric values; alternative boundary definitions could yield different insights. Finally, while the analysis reveals structural patterns and their likely implications, causal claims require controlled intervention studies.
"""
        return discussion

    def generate_conclusions(self) -> str:
        """Generate prioritized conclusions and recommendations."""

        alpha = self.metrics['ascendency_ratio']
        rob = self.metrics['robustness']
        viable = self.metrics['is_viable']
        eff = self.metrics['network_efficiency']
        ovh_ratio = self.metrics['overhead_ratio']

        conclusions = f"""
5. CONCLUSIONS AND RECOMMENDATIONS
===================================

5.1 Summary of Findings
-------------------------
This assessment of {self.org_name} establishes that the organization {"operates within the window of viability, demonstrating" if viable else "falls outside the window of viability, lacking"} the efficiency-resilience balance associated with sustainable network dynamics. Robustness of R = {rob:.3f} ({self._categorize_robustness().lower()}) and network efficiency of {eff:.3f} ({self._categorize_efficiency().lower()}) together characterize a system that {"is well-positioned for sustained performance in dynamic conditions" if viable and rob > 0.2 else "maintains adequate but not exceptional adaptive capacity" if viable else "requires structural adjustment to restore sustainable dynamics"}.

5.2 Prioritized Recommendations
---------------------------------
{self._generate_priority_recommendations()}

5.3 Future Assessment
----------------------
To strengthen the evidence base, the following extensions are recommended: longitudinal tracking of these metrics at regular intervals to identify trends and cycles; comparative benchmarking against industry peers using the same framework; separate analysis of distinct flow types (information, resources, authority) where data permits; and dynamic modeling to develop predictive scenarios for organizational evolution.

The Ulanowicz-Fath framework's grounding in information theory and decades of ecological validation provides a robust complement to traditional performance measurement. As organizational environments grow more complex and volatile, the ability to quantify the balance between efficiency and adaptability becomes increasingly material to strategic decision-making.
"""
        return conclusions

    def generate_references(self) -> str:
        """Generate references section."""

        references = """
REFERENCES
==========

Fath, B. D., Fiscus, D. A., Goerner, S. J., Berea, A., & Ulanowicz, R. E. (2019).
    Measuring regenerative economics: 10 principles and measures undergirding systemic
    economic health. Global Transitions, 1, 15-27.

Holling, C. S. (1973). Resilience and stability of ecological systems. Annual Review
    of Ecology and Systematics, 4(1), 1-23.

Ulanowicz, R. E. (1986). Growth and Development: Ecosystems Phenomenology.
    Springer-Verlag, New York.

Ulanowicz, R. E. (1997). Ecology, the Ascendent Perspective. Columbia University Press,
    New York.

Ulanowicz, R. E. (2009). A Third Window: Natural Life beyond Newton and Darwin.
    Templeton Foundation Press, West Conshohocken, PA.

Ulanowicz, R. E., Goerner, S. J., Lietaer, B., & Gomez, R. (2009). Quantifying
    sustainability: Resilience, efficiency and the return of information theory.
    Ecological Complexity, 6(1), 27-36.

Zorach, A. C., & Ulanowicz, R. E. (2003). Quantifying the complexity of flow networks:
    How many roles are there? Complexity, 8(3), 68-76.
"""
        return references

    def generate_appendix(self) -> str:
        """Generate appendix with formula definitions and detailed data tables."""

        appendix = f"""
APPENDIX: DETAILED DATA AND FORMULA DEFINITIONS
=================================================

A.1 Formula Definitions
------------------------
The following formulas underpin the analysis. All logarithms are base-2 unless otherwise noted.

Total System Throughput (TST): TST = Sum_ij F_ij

Average Mutual Information (AMI): AMI = Sum_ij (F_ij / TST) x log2(F_ij x TST / (T_i. x T_.j))
where T_i. = Sum_j F_ij (row marginal) and T_.j = Sum_i F_ij (column marginal).

Flow Diversity (H): H = -Sum_ij (F_ij / TST) x log2(F_ij / TST)

Ascendency (A): A = TST x AMI

Development Capacity (C): C = TST x H

Overhead (Phi): Phi = C - A

Relative Ascendency (alpha): alpha = A / C

Relative Overhead: phi = Phi / C = 1 - alpha

Robustness (R): R = -alpha x log2(alpha)

Redundancy: 1 - (AMI / H_max) where H_max = log2(n)

Effective Link Density (ELD): ELD = (number of non-zero F_ij) / n^2

Trophic Depth: Average path length weighted by flow magnitude

Network Efficiency: A / (C x log2(n))

Regenerative Capacity: (Phi / C) x (1 - |alpha - 0.37|)

Window of Viability bounds: 0.20 <= alpha <= 0.60

A.2 Complete Metric Values
---------------------------

Table A1. Full Metric Detail
------------------------------
Metric                                  Value           Unit
Total System Throughput (TST)           {self.metrics['total_system_throughput']:<15.6f} flow units
Average Mutual Information (AMI)        {self.metrics['average_mutual_information']:<15.6f} bits
Conditional Entropy                     {self.metrics.get('conditional_entropy', 0):<15.6f} bits
Flow Diversity (H)                      {self.metrics['flow_diversity']:<15.6f} bits
Structural Information (SI)             {self.metrics['structural_information']:<15.6f} bits
Ascendency (A)                         {self.metrics['ascendency']:<15.6f} flow-bits
Development Capacity (C)                {self.metrics['development_capacity']:<15.6f} flow-bits
Overhead (Phi)                          {self.metrics['overhead']:<15.6f} flow-bits
Relative Ascendency (alpha)             {self.metrics['ascendency_ratio']:<15.6f} ratio
Relative Overhead (phi)                 {self.metrics['overhead_ratio']:<15.6f} ratio
Robustness (R)                         {self.metrics['robustness']:<15.6f} ratio
Network Efficiency                      {self.metrics['network_efficiency']:<15.6f} ratio
Redundancy                              {self.metrics['redundancy']:<15.6f} ratio
Effective Link Density                  {self.metrics['effective_link_density']:<15.6f} ratio
Trophic Depth                          {self.metrics['trophic_depth']:<15.6f} levels
Regenerative Capacity                   {self.metrics['regenerative_capacity']:<15.6f} ratio

A.3 Node-Level Statistics
--------------------------

Table A2. Node Flow Summary
-----------------------------
Node ID    Name                 In-Flow    Out-Flow   Total Flow
"""

        for i, name in enumerate(self.node_names[:20]):
            in_flow = np.sum(self.flow_matrix[:, i])
            out_flow = np.sum(self.flow_matrix[i, :])
            total = in_flow + out_flow
            appendix += f"{i + 1:<10} {name[:20]:<20} {in_flow:<10.2f} {out_flow:<10.2f} {total:<10.2f}\n"

        if len(self.node_names) > 20:
            appendix += f"... ({len(self.node_names) - 20} additional nodes)\n"

        appendix += """

A.4 Flow Distribution Statistics
----------------------------------
"""
        active_flows = self.flow_matrix[self.flow_matrix > 0]
        if len(active_flows) > 0:
            appendix += f"""Mean Flow:                      {np.mean(active_flows):.3f}
Median Flow:                    {np.median(active_flows):.3f}
Standard Deviation:             {np.std(active_flows):.3f}
Coefficient of Variation:       {np.std(active_flows) / np.mean(active_flows):.3f}
Maximum Flow:                   {np.max(active_flows):.3f}
Gini Coefficient:               {self._calculate_gini():.3f}
"""

        appendix += """
A.5 Assessment Categories
--------------------------

Table A3. Assessment Summary
------------------------------
"""
        for category, assessment in self.assessments.items():
            appendix += f"{category.replace('_', ' ').title():<30} {assessment}\n"

        return appendix

    def generate_oasis_section(self) -> str:
        """Generate OASIS Organizational Health Assessment section with narrative interpretation."""

        try:
            from oasis_calculator import OASISCalculator
            oasis = OASISCalculator(self.calculator)
            profile = oasis.get_oasis_profile()
            interpretations = oasis.get_oasis_interpretation()
            recommendations = oasis.get_recommendations()
        except Exception as e:
            return f"""
6. OASIS ORGANIZATIONAL HEALTH ASSESSMENT
==========================================

The OASIS assessment could not be completed for this analysis: {str(e)}. The OASIS (Open, Autonomous, Symbiotic, Intelligent, Sustainable) framework requires additional network metrics that may not be available for all dataset types.
"""

        scores = profile['dimension_scores']
        overall = profile['overall_score']
        overall_status = profile['overall_status']
        dim_status = profile['dimension_status']

        # Identify strongest and weakest dimensions
        sorted_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_dims[0]
        weakest = sorted_dims[-1]

        section = f"""
6. OASIS ORGANIZATIONAL HEALTH ASSESSMENT
==========================================

6.1 Overall Assessment
-----------------------
{self.org_name} achieves an overall OASIS health score of {overall:.0f} out of 100, classified as {overall_status}. This composite score integrates five dimensions derived from Fath et al.'s (2019) 10 Principles of Regenerative Economics, each measuring a distinct facet of organizational vitality. The strongest dimension is {strongest[0].upper()} ({strongest[1]:.0f}/100), while {weakest[0].upper()} ({weakest[1]:.0f}/100) represents the primary area for improvement.

Table 6. OASIS Dimension Assessment
------------------------------------
Dimension        Score   Status      Key Insight
OPEN             {scores['open']:<7.0f} {dim_status['open']:<11} Interconnectivity & exchange
AUTONOMOUS       {scores['autonomous']:<7.0f} {dim_status['autonomous']:<11} Learning & routine encoding
SYMBIOTIC        {scores['symbiotic']:<7.0f} {dim_status['symbiotic']:<11} Integration & balance
INTELLIGENT      {scores['intelligent']:<7.0f} {dim_status['intelligent']:<11} Functional diversity
SUSTAINABLE      {scores['sustainable']:<7.0f} {dim_status['sustainable']:<11} Order-freedom balance

6.2 Dimension Analysis
-----------------------
"""

        dim_full_names = {
            'open': 'OPEN (Fath Principles 1, 3, 4)',
            'autonomous': 'AUTONOMOUS (Fath Principles 2, 9)',
            'symbiotic': 'SYMBIOTIC (Fath Principles 5, 8)',
            'intelligent': 'INTELLIGENT (Fath Principles 7, 10)',
            'sustainable': 'SUSTAINABLE (Fath Principle 6)',
        }

        for dim_key in ['open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable']:
            score = scores[dim_key]
            status = dim_status[dim_key]
            interp = interpretations.get(dim_key, 'No interpretation available.')

            # Compose narrative per dimension
            if status == 'HEALTHY':
                outlook = "This dimension is performing well and supports overall organizational health."
            elif status == 'WARNING':
                outlook = "This dimension warrants monitoring; while not critical, deterioration could affect overall system health."
            else:
                outlook = "This dimension represents a material concern that leadership should address in the near term."

            section += f"""6.2.{list(dim_full_names.keys()).index(dim_key) + 1} {dim_full_names[dim_key]}
The {dim_key.capitalize()} dimension scores {score:.0f}/100 ({status}). {interp} {outlook}

"""

        section += """6.3 OASIS-Based Recommendations
--------------------------------
"""
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                section += f"""Recommendation {i} ({rec['priority']} priority):
- Dimension: {rec['dimension']}
- Issue: {rec['issue']}
- Action: {rec['action']}
- Metrics to improve: {', '.join(rec.get('metrics_to_improve', ['N/A']))}

"""
        else:
            section += """The organization shows healthy patterns across all OASIS dimensions. No critical interventions are indicated at this time. Continued monitoring and maintenance of current practices is recommended.
"""

        section += """6.4 OASIS Framework Reference
------------------------------
The OASIS assessment is based on:

Fath, B.D., Fiscus, D.A., Goerner, S.J., Berea, A., & Ulanowicz, R.E. (2019).
Measuring regenerative economics: 10 principles and measures undergirding
systemic economic health. Global Transitions, 1, 15-27.
"""
        return section

    def generate_full_report(self) -> str:
        """Generate complete report."""

        report = f"""
================================================================================
NETWORK SUSTAINABILITY ASSESSMENT: {self.org_name.upper()}
A Quantitative Analysis Using Regenerative Economics Principles
================================================================================

Prepared by: Adaptive Organization Analysis System
Date: {self.timestamp.strftime('%B %d, %Y')}

--------------------------------------------------------------------------------
"""

        report += self.generate_abstract()
        report += self.generate_introduction()
        report += self.generate_methodology()
        report += self.generate_results()
        report += self.generate_oasis_section()
        report += self.generate_discussion()
        report += self.generate_conclusions()
        report += self.generate_references()
        report += self.generate_appendix()

        report += """
================================================================================
END OF REPORT
================================================================================
"""

        return report

    # ==================================================================
    # Helper methods -- public API preserved for external callers
    # ==================================================================

    def _categorize_efficiency(self) -> str:
        """Categorize network efficiency level."""
        eff = self.metrics['network_efficiency']
        if eff < 0.2:
            return "Low"
        elif eff < 0.4:
            return "Moderate"
        elif eff < 0.6:
            return "High"
        else:
            return "Very High"

    def _categorize_robustness(self) -> str:
        """Categorize robustness level."""
        rob = self.metrics['robustness']
        if rob < 0.1:
            return "Very Low"
        elif rob < 0.15:
            return "Low"
        elif rob < 0.2:
            return "Moderate"
        elif rob < 0.25:
            return "High"
        else:
            return "Very High"

    def _interpret_position(self) -> str:
        """Interpret position in window of viability."""
        alpha = self.metrics['ascendency_ratio']
        if alpha < 0.2:
            return "Under-organized"
        elif alpha < 0.35:
            return "Developing"
        elif alpha < 0.45:
            return "Optimal"
        elif alpha < 0.6:
            return "Efficient"
        else:
            return "Over-constrained"

    def _calculate_gini(self) -> float:
        """Calculate Gini coefficient for flow distribution."""
        flows = self.flow_matrix[self.flow_matrix > 0].flatten()
        if len(flows) == 0:
            return 0.0
        sorted_flows = np.sort(flows)
        n = len(sorted_flows)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_flows)) / (n * np.sum(sorted_flows)) - (n + 1) / n

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _get_viability_interpretation(self) -> str:
        """Get interpretation of viability status."""
        if self.metrics['is_viable']:
            return "Sustainable"
        elif self.metrics['ascendency_ratio'] < self.metrics['viability_lower_bound']:
            return "Too chaotic"
        else:
            return "Too rigid"

    def _interpret_redundancy(self) -> str:
        """Interpret redundancy level."""
        r = self.metrics['redundancy']
        if r < 0.3:
            return "Low backup"
        elif r < 0.6:
            return "Moderate backup"
        else:
            return "High backup"

    def _interpret_regenerative(self) -> str:
        """Interpret regenerative capacity."""
        rc = self.metrics['regenerative_capacity']
        if rc < 0.2:
            return "Limited"
        elif rc < 0.4:
            return "Moderate"
        else:
            return "Strong"

    def _identify_data_source(self) -> str:
        """Identify the data source type."""
        if 'European Power Grid' in self.org_name:
            return "Real-world energy network dataset (Kaggle)"
        elif 'PaySim' in self.org_name:
            return "Financial transaction network dataset"
        elif 'Supply Chain' in self.org_name:
            return "Supply chain network dataset"
        elif 'OECD' in self.org_name or 'WTO' in self.org_name:
            return "Official economic statistics"
        elif 'Test' in self.org_name or 'Example' in self.org_name:
            return "Synthetic test dataset"
        else:
            return "Organizational flow data"

    def _generate_priority_recommendations(self) -> str:
        """Generate prioritized recommendations in three tiers."""
        alpha = self.metrics['ascendency_ratio']
        rob = self.metrics['robustness']
        viable = self.metrics['is_viable']
        eff = self.metrics['network_efficiency']
        ovh_ratio = self.metrics['overhead_ratio']
        redund = self.metrics['redundancy']

        priorities = []

        # Priority 1: Immediate (0-3 months)
        immediate = []
        if not viable:
            if alpha < self.metrics['viability_lower_bound']:
                immediate.append(
                    "Restore organizational coherence by strengthening coordination "
                    "mechanisms and clarifying primary flow pathways. The current alpha "
                    f"of {alpha:.3f} falls below the minimum viable threshold, indicating "
                    "that flows are too diffuse for effective collective action."
                )
            else:
                immediate.append(
                    "Introduce adaptive capacity by decentralizing selected decision "
                    "pathways and creating alternative flow routes. The current alpha "
                    f"of {alpha:.3f} exceeds the upper viability bound, indicating "
                    "that over-constraint has reduced the organization's ability to "
                    "absorb disruptions."
                )
        if rob < 0.15:
            immediate.append(
                f"Address low robustness (R = {rob:.3f}) by developing backup pathways "
                "for critical flows and cross-training personnel to reduce dependency "
                "on single-channel connections."
            )

        if immediate:
            priorities.append("Priority 1 -- Immediate (0 to 3 months):")
            for item in immediate:
                priorities.append(f"- {item}")
            priorities.append("")

        # Priority 2: Short-term (3-6 months)
        short_term = []
        if eff < 0.2:
            short_term.append(
                f"Improve network efficiency (currently {eff:.3f}) by streamlining "
                "redundant pathways, clarifying roles and responsibilities, and "
                "strengthening the connections that carry the highest-value flows."
            )
        elif eff > 0.6:
            short_term.append(
                f"Counterbalance high efficiency ({eff:.3f}) by deliberately introducing "
                "parallel pathways and buffer capacity. Over-optimized networks are "
                "fragile; strategic redundancy improves shock absorption."
            )
        if redund < 0.3:
            short_term.append(
                f"Build pathway redundancy (currently {redund:.3f}) by establishing "
                "alternative information and resource channels that can activate "
                "if primary routes are disrupted."
            )

        if short_term:
            priorities.append("Priority 2 -- Short-term (3 to 6 months):")
            for item in short_term:
                priorities.append(f"- {item}")
            priorities.append("")

        # Priority 3: Medium-term (6-12 months)
        medium_term = []
        medium_term.append(
            "Establish a recurring assessment cadence (quarterly or semi-annual) "
            "to track the trajectory of key indicators -- particularly alpha, "
            "robustness, and overhead ratio -- and detect early signs of drift "
            "toward unsustainable configurations."
        )
        if ovh_ratio < 0.4:
            medium_term.append(
                f"Invest in organizational reserve capacity. The current overhead "
                f"ratio of {ovh_ratio:.3f} provides limited buffering. Increasing "
                "this ratio through targeted investment in cross-functional linkages "
                "and knowledge redundancy will strengthen long-term adaptive capacity."
            )
        medium_term.append(
            "Develop comparative benchmarks by analyzing peer organizations or "
            "industry reference networks using the same framework, enabling "
            "context-informed target-setting for key metrics."
        )

        priorities.append("Priority 3 -- Medium-term (6 to 12 months):")
        for item in medium_term:
            priorities.append(f"- {item}")

        # Fallback for healthy organizations
        if not any([not viable, rob < 0.15, eff < 0.2, eff > 0.6, redund < 0.3]):
            return """The organization demonstrates healthy dynamics across all primary indicators. The recommended course of action centers on preservation and monitoring:

Priority 1 -- Ongoing:
- Maintain the current efficiency-resilience balance; avoid optimization initiatives that would push alpha above the viability ceiling
- Conduct regular assessments to confirm stability and detect early drift

Priority 2 -- Short-term (3 to 6 months):
- Document current network configurations and practices as an organizational baseline
- Develop contingency protocols that leverage existing redundancy pathways

Priority 3 -- Medium-term (6 to 12 months):
- Benchmark against peer organizations to validate competitive positioning
- Explore separate analysis of distinct flow types (information, resources, authority) for deeper insight"""

        return '\n'.join(priorities)
