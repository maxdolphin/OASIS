"""
Centralized Documentation Registry for the Adaptive Organization UI.

Single source of truth for all tooltips, explanations, citations, and anchors.
Every visible UI element maps to an entry here. The registry drives:
  - Tier 1: Inline tooltips (short plain-language explanation)
  - Tier 2: In-app documentation panel (full definition, interpretation, citation)
  - Tier 3: External reference links (DOI / open-access URLs)

Usage:
    from src.docs_registry import DOCS, info_button
    info_button("tst")  # renders ⓘ icon with tooltip + link to #metric-tst
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------
CAT_CORE = "Core Ulanowicz Metrics"
CAT_INFO = "Information Theory Metrics"
CAT_SUSTAIN = "Sustainability Assessment"
CAT_REGEN = "Regenerative Economics"
CAT_NETWORK = "Network Structure & Topology"
CAT_OASIS = "OASIS Dimensions"
CAT_ROLES = "Roles & Functional Specialization"
CAT_FLOW = "Flow Statistics"
CAT_ROBUST = "Robustness & Resilience"
CAT_CENTRALITY = "Centrality Measures"
CAT_COMMUNITY = "Community Structure"
CAT_EXTENDED = "Extended Network Metrics"
CAT_VIZ = "Visualizations"
CAT_HEALTH = "Health Assessment Labels"
CAT_BALANCE = "Balance Indicators"
CAT_ECOSYSTEM = "Ecosystem Boundary Metrics"

# ---------------------------------------------------------------------------
# Primary scientific references (reused across entries)
# ---------------------------------------------------------------------------
REF_ULANOWICZ_2009 = (
    "Ulanowicz, R.E., Goerner, S.J., Lietaer, B., & Gomez, R. (2009). "
    "\"Quantifying sustainability: Resilience, efficiency and the return of "
    "information theory.\" Ecological Complexity, 6(1), 27-36."
)
REF_FATH_2019 = (
    "Fath, B.D., Fiscus, D.A., Goerner, S.J., Berea, A., & Ulanowicz, R.E. "
    "(2019). \"Measuring regenerative economics: 10 principles and measures "
    "undergirding systemic economic health.\" Global Transitions, 1, 15-27."
)
REF_ZORACH_2003 = (
    "Zorach, A.C. & Ulanowicz, R.E. (2003). \"Quantifying the complexity of "
    "flow networks: How many roles are there?\" Complexity, 8(3), 68-76."
)
REF_FINN_1976 = (
    "Finn, J.T. (1976). \"Measures of ecosystem structure and function derived "
    "from analysis of flows.\" Journal of Theoretical Biology, 56, 363-380."
)
REF_HOLLING_1973 = (
    "Holling, C.S. (1973). \"Resilience and stability of ecological systems.\" "
    "Annual Review of Ecology and Systematics, 4(1), 1-23."
)
REF_LINDEMAN_1942 = (
    "Lindeman, R.L. (1942). \"The trophic-dynamic aspect of ecology.\" "
    "Ecology, 23(4), 399-418."
)
REF_WATTS_STROGATZ_1998 = (
    "Watts, D.J. & Strogatz, S.H. (1998). \"Collective dynamics of "
    "'small-world' networks.\" Nature, 393(6684), 440-443."
)
DOI_ULANOWICZ_2009 = "https://doi.org/10.1016/j.ecocom.2008.10.005"
DOI_FATH_2019 = "https://doi.org/10.1016/j.glt.2019.06.002"
DOI_ZORACH_2003 = "https://doi.org/10.1002/cplx.10028"

# ---------------------------------------------------------------------------
# Registry: dict[str, dict]
#   key        – unique identifier used by info_button(key)
#   tooltip    – 1-2 sentence plain-language explanation (Tier 1)
#   definition – formal definition (Tier 2)
#   interpret  – what high / low values mean in organizational context
#   formula    – LaTeX-style formula string (optional)
#   citation   – full reference string
#   doi        – DOI or URL (optional, Tier 3)
#   oasis_map  – which OASIS dimension(s) this feeds into (optional)
#   category   – grouping for the documentation page
#   anchor     – HTML anchor id (auto-generated if omitted)
# ---------------------------------------------------------------------------

DOCS: Dict[str, Dict[str, Any]] = {

    # =====================================================================
    # CORE ULANOWICZ METRICS
    # =====================================================================
    "tst": {
        "label": "Total System Throughput (TST)",
        "tooltip": (
            "The sum of all flows in the network — the total metabolic "
            "activity of the organization."
        ),
        "definition": (
            "Total System Throughput (TST) is the sum of every flow in the "
            "network matrix. It represents the overall size or metabolism of "
            "the system, analogous to GDP for an economy or gross primary "
            "production for an ecosystem."
        ),
        "interpret": (
            "**High TST**: The organization processes large volumes of "
            "resources, information, or value. "
            "**Low TST**: Limited overall activity. "
            "TST alone does not indicate health — a large, inefficient "
            "system can have high TST."
        ),
        "formula": "TST = \\sum_{i,j} T_{ij}",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_CORE,
    },

    "ascendency": {
        "label": "Ascendency (A)",
        "tooltip": (
            "Organized power of the system — how much of the network's "
            "activity is channeled into efficient, constrained pathways."
        ),
        "definition": (
            "Ascendency quantifies the degree to which flows are organized "
            "into definite, constrained pathways. It combines system size "
            "(TST) with the mutual information of the flow structure. "
            "Higher ascendency means more directed, purposeful flows."
        ),
        "interpret": (
            "**High A**: Efficient, streamlined operations with clear "
            "resource routing. Risk: brittleness if too high. "
            "**Low A**: Disorganized flows, lack of clear structure."
        ),
        "formula": "A = \\sum_{i,j} T_{ij} \\cdot \\ln\\!\\left(\\frac{T_{ij} \\cdot T_{\\cdot\\cdot}}{T_{i\\cdot} \\cdot T_{\\cdot j}}\\right)",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Sustainable",
        "category": CAT_CORE,
    },

    "overhead": {
        "label": "Overhead / Reserve (Φ)",
        "tooltip": (
            "The system's flexibility and reserve capacity — redundant "
            "pathways that provide resilience under stress."
        ),
        "definition": (
            "Overhead (Φ) is the portion of Development Capacity not "
            "expressed as Ascendency. It represents the system's reserves: "
            "redundant pathways, unused capacity, and alternative routes "
            "that activate under stress."
        ),
        "interpret": (
            "**High Φ**: Lots of backup pathways — resilient but potentially "
            "wasteful. "
            "**Low Φ**: Lean operations with few alternatives — efficient "
            "but fragile."
        ),
        "formula": "\\Phi = C - A",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Sustainable",
        "category": CAT_CORE,
    },

    "capacity": {
        "label": "Development Capacity (C)",
        "tooltip": (
            "Total potential of the system for both organization and "
            "flexibility — the upper bound on what the system can achieve."
        ),
        "definition": (
            "Development Capacity (C) is the scaled system indeterminacy "
            "and represents the maximum possible complexity. It equals "
            "Ascendency plus Overhead (C = A + Φ) and sets an upper "
            "bound on how organized the network can become."
        ),
        "interpret": (
            "**High C**: Large total capacity for development and change. "
            "**Low C**: Limited room for growth or adaptation."
        ),
        "formula": "C = -\\sum_{i,j} T_{ij} \\cdot \\ln\\!\\left(\\frac{T_{ij}}{T_{\\cdot\\cdot}}\\right) = A + \\Phi",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_CORE,
    },

    "relative_ascendency": {
        "label": "Relative Ascendency (α = A/C)",
        "tooltip": (
            "The fraction of system capacity realized as organized structure — "
            "the key sustainability indicator. Optimal near 0.37."
        ),
        "definition": (
            "Relative Ascendency (α) is the ratio of Ascendency to "
            "Development Capacity. It is the single most important "
            "sustainability metric: it tells you what fraction of the "
            "system's potential is dedicated to organized, efficient "
            "operation versus kept as flexible reserve."
        ),
        "interpret": (
            "**α ≈ 0.37**: Optimal balance — maximum robustness. "
            "**α > 0.6**: Over-organized — brittle, unable to adapt. "
            "**α < 0.2**: Under-organized — chaotic, lacking structure. "
            "**0.2 ≤ α ≤ 0.6**: Within the Window of Viability."
        ),
        "formula": "\\alpha = A / C",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Sustainable",
        "category": CAT_SUSTAIN,
    },

    "robustness": {
        "label": "Robustness (R)",
        "tooltip": (
            "The system's ability to withstand disturbances — peaks at "
            "α ≈ 0.368 (1/e), maximum value ≈ 0.368."
        ),
        "definition": (
            "Robustness measures how well the system can absorb shocks "
            "and maintain function. It is derived from the relative "
            "ascendency using the formula R = -α·ln(α). This function "
            "has a single maximum at α = 1/e ≈ 0.368."
        ),
        "interpret": (
            "**R near 0.368**: Maximum resilience — ideal balance. "
            "**R near 0**: System is either too rigid (α→1) or too "
            "chaotic (α→0) to absorb perturbations."
        ),
        "formula": "R = -\\alpha \\cdot \\ln(\\alpha)",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Sustainable",
        "category": CAT_SUSTAIN,
    },

    "ami": {
        "label": "Average Mutual Information (AMI)",
        "tooltip": (
            "How organized the flow patterns are — the information content "
            "that distinguishes actual flows from random ones."
        ),
        "definition": (
            "Average Mutual Information quantifies the degree of constraint "
            "or organization in the network's flow pattern. It measures "
            "how much knowing where a flow comes from tells you where it "
            "goes (and vice versa)."
        ),
        "interpret": (
            "**High AMI**: Flows follow predictable, structured routes — "
            "departments have clear, dedicated channels. "
            "**Low AMI**: Flows are diffuse and unpredictable."
        ),
        "formula": "AMI = \\frac{1}{TST}\\sum_{i,j} T_{ij} \\cdot \\ln\\!\\left(\\frac{T_{ij} \\cdot T_{\\cdot\\cdot}}{T_{i\\cdot} \\cdot T_{\\cdot j}}\\right)",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Autonomous",
        "category": CAT_INFO,
    },

    "flow_diversity": {
        "label": "Flow Diversity (H)",
        "tooltip": (
            "Shannon entropy of the flow distribution — how evenly "
            "resources are spread across all connections."
        ),
        "definition": (
            "Flow Diversity is the Shannon entropy of the normalized flow "
            "matrix. It measures how evenly flows are distributed across "
            "all network connections. Maximum when all flows are equal; "
            "minimum when all flow goes through a single link."
        ),
        "interpret": (
            "**High H**: Resources spread broadly — no single pathway "
            "dominates. "
            "**Low H**: Resources concentrated in a few dominant channels."
        ),
        "formula": "H = -\\sum_{i,j} \\frac{T_{ij}}{TST} \\cdot \\ln\\!\\left(\\frac{T_{ij}}{TST}\\right)",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Open",
        "category": CAT_INFO,
    },

    "conditional_entropy": {
        "label": "Conditional Entropy (Hc)",
        "tooltip": (
            "Remaining uncertainty after accounting for organized flows — "
            "the system's inherent flexibility."
        ),
        "definition": (
            "Conditional Entropy is the portion of Flow Diversity not "
            "explained by organized structure (AMI). It represents the "
            "residual freedom or flexibility in the system: Hc = H - AMI."
        ),
        "interpret": (
            "**High Hc**: The system retains significant flexibility and "
            "choice in how resources can be rerouted. "
            "**Low Hc**: Flows are tightly constrained with little room "
            "for adaptation."
        ),
        "formula": "H_c = H - AMI",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Intelligent",
        "category": CAT_INFO,
    },

    "redundancy": {
        "label": "Redundancy (Φ/C)",
        "tooltip": (
            "Proportion of system capacity kept as backup pathways — "
            "the complement of relative ascendency."
        ),
        "definition": (
            "Redundancy is the ratio of Overhead to Development Capacity "
            "(Φ/C = 1 - α). It quantifies the fraction of system potential "
            "maintained as alternative pathways and reserve capacity."
        ),
        "interpret": (
            "**High redundancy (>0.8)**: Excessive backup — wasteful, "
            "under-organized. "
            "**Low redundancy (<0.4)**: Lean but fragile — few alternatives. "
            "**Optimal (0.4–0.65)**: Good balance of efficiency and resilience."
        ),
        "formula": "Redundancy = \\Phi / C = 1 - \\alpha",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_SUSTAIN,
    },

    "structural_information": {
        "label": "Structural Information (SI)",
        "tooltip": (
            "How constrained the network structure is compared to a "
            "maximally random network of the same size."
        ),
        "definition": (
            "Structural Information measures the degree of constraint in "
            "the network topology, independent of flow magnitudes. "
            "SI = ln(n²) - H, where n is the number of nodes."
        ),
        "interpret": (
            "**High SI**: Network has a strongly defined structure with "
            "clear roles and hierarchy. "
            "**Low SI**: Network resembles a random graph."
        ),
        "formula": "SI = \\ln(n^2) - H",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_EXTENDED,
    },

    "effective_link_density": {
        "label": "Effective Link Density",
        "tooltip": (
            "Flow-weighted connectivity — how many connections are "
            "actually doing meaningful work."
        ),
        "definition": (
            "Effective Link Density combines topological connectivity "
            "with information-theoretic organization. It weights the "
            "fraction of active links by AMI to distinguish networks "
            "with many weak links from those with fewer but meaningful ones."
        ),
        "interpret": (
            "**High**: Dense, meaningful connections. "
            "**Low**: Sparse or superficial connectivity."
        ),
        "formula": "ELD = \\frac{L_{active}}{L_{max}} \\cdot \\frac{AMI}{AMI_{max}}",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_EXTENDED,
    },

    "trophic_depth": {
        "label": "Trophic Depth",
        "tooltip": (
            "Average number of hierarchical levels in the network — "
            "how deep the organizational chain runs."
        ),
        "definition": (
            "Trophic Depth is the average shortest path length (in hops, "
            "not flow magnitude) through the network. It indicates the "
            "depth of the organizational hierarchy, analogous to trophic "
            "levels in ecology."
        ),
        "interpret": (
            "**High (>3)**: Deep hierarchy with many intermediate layers. "
            "**Low (1–2)**: Flat organization with direct connections. "
            "Real-world networks typically range from 1 to 10."
        ),
        "formula": "TD = \\langle l \\rangle \\text{ (avg. shortest path, unweighted)}",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_EXTENDED,
    },

    "finn_cycling_index": {
        "label": "Finn Cycling Index (FCI)",
        "tooltip": (
            "Fraction of total throughput involved in cycling — measures "
            "internal resource recycling and feedback loops."
        ),
        "definition": (
            "The Finn Cycling Index quantifies how much of the total system "
            "throughput is recycled within the network through loops and "
            "feedback cycles. It detects self-loops and bidirectional "
            "2-node cycles."
        ),
        "interpret": (
            "**FCI > 0.5**: Strong internal recycling — knowledge and "
            "resources circulate well. "
            "**FCI < 0.1**: Little recycling — one-way, extractive flows. "
            "Healthy organizations typically show moderate cycling (0.1–0.5)."
        ),
        "formula": "FCI = \\frac{\\text{cycling flow}}{TST}",
        "citation": REF_FINN_1976,
        "oasis_map": "Autonomous",
        "category": CAT_REGEN,
    },

    "network_efficiency": {
        "label": "Network Efficiency",
        "tooltip": (
            "How well the network converts its capacity into organized "
            "throughput — equivalent to relative ascendency (α)."
        ),
        "definition": (
            "Network Efficiency equals the Relative Ascendency (α = A/C). "
            "It measures what fraction of the system's total capacity is "
            "realized as organized, directed activity."
        ),
        "interpret": (
            "**High (>0.6)**: Very efficient but potentially brittle. "
            "**Low (<0.2)**: Under-utilizing network capacity. "
            "**Optimal (0.35–0.40)**: Peak sustainability zone."
        ),
        "formula": "Efficiency = \\alpha = A / C",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_SUSTAIN,
    },

    "regenerative_capacity": {
        "label": "Regenerative Capacity",
        "tooltip": (
            "The system's ability to self-renew and adapt — highest "
            "when efficiency ratio is near the optimal 0.37."
        ),
        "definition": (
            "Regenerative Capacity combines robustness with proximity "
            "to the optimal efficiency ratio. It measures the system's "
            "potential for self-renewal and adaptation."
        ),
        "interpret": (
            "**High**: Strong self-renewal — the organization can "
            "regenerate after disturbances. "
            "**Low**: Limited adaptive capacity — recovery is slow or "
            "unlikely."
        ),
        "formula": "RC = R \\cdot (1 - |\\alpha - 0.37|)",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Sustainable",
        "category": CAT_REGEN,
    },

    "fitness": {
        "label": "Fitness for Evolution (F)",
        "tooltip": (
            "Evolutionary fitness based on the efficiency-resilience "
            "balance — maximum at α ≈ 0.46 for β = 1.288."
        ),
        "definition": (
            "Fitness for Evolution extends robustness with an empirical "
            "exponent β = 1.288 derived from ecosystem studies. It "
            "captures a system's ability to persist and evolve over time."
        ),
        "interpret": (
            "**High F**: The organization is well-positioned for "
            "long-term evolutionary persistence. "
            "**Low F**: The system may not survive environmental shifts."
        ),
        "formula": "F = -e \\cdot \\alpha^{\\beta} \\cdot \\ln(\\alpha^{\\beta}), \\quad \\beta = 1.288",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_SUSTAIN,
    },

    # =====================================================================
    # WINDOW OF VIABILITY
    # =====================================================================
    "window_of_viability": {
        "label": "Window of Viability",
        "tooltip": (
            "The sustainable operating zone: Ascendency must fall between "
            "20% and 60% of Development Capacity."
        ),
        "definition": (
            "The Window of Viability defines the range of Relative "
            "Ascendency (α) within which a system can sustain itself. "
            "Below α = 0.2 the system is too chaotic; above α = 0.6 "
            "it is too rigid. The optimal zone is α = 0.35–0.40, where "
            "robustness peaks."
        ),
        "interpret": (
            "**Inside window (0.2–0.6)**: System is viable — it has "
            "enough structure to function and enough flexibility to adapt. "
            "**Outside window**: Unsustainable — either over-optimized "
            "(brittle) or under-organized (chaotic)."
        ),
        "formula": "0.2 \\cdot C \\leq A \\leq 0.6 \\cdot C",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "oasis_map": "Sustainable",
        "category": CAT_SUSTAIN,
    },

    "viable_system": {
        "label": "Viable System",
        "tooltip": (
            "Whether the organization falls inside the Window of "
            "Viability — the sustainable operating zone (α between 0.2 and 0.6)."
        ),
        "definition": (
            "A system is 'viable' when its Relative Ascendency (α) falls "
            "within the Window of Viability: 0.2 ≤ α ≤ 0.6. This means "
            "the balance between efficiency and redundancy is sustainable."
        ),
        "interpret": (
            "**Yes / ✅**: The organization is in a sustainable operating "
            "zone. "
            "**No / ❌**: The organization needs rebalancing — it is either "
            "too rigid or too chaotic."
        ),
        "formula": "Viable \\iff 0.2 \\leq \\alpha \\leq 0.6",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_SUSTAIN,
    },

    "distance_from_optimum": {
        "label": "Distance from Optimum",
        "tooltip": (
            "How far the current efficiency ratio (α) is from the "
            "theoretical optimum of 0.37."
        ),
        "definition": (
            "The absolute distance between the system's Relative "
            "Ascendency (α) and the theoretical optimum value of 0.37, "
            "where robustness R = -α·ln(α) reaches its maximum."
        ),
        "interpret": (
            "**Near 0**: Organization is at or near optimal balance. "
            "**> 0.1**: Significant deviation — system would benefit from "
            "rebalancing toward the optimum."
        ),
        "formula": "|\\alpha - 0.37|",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_SUSTAIN,
    },

    # =====================================================================
    # OASIS DIMENSIONS
    # =====================================================================
    "oasis_overall": {
        "label": "OASIS Overall Health Score",
        "tooltip": (
            "Weighted average of all five OASIS dimensions — the "
            "organization's overall ecosystemic health (0–100)."
        ),
        "definition": (
            "The OASIS (Open, Autonomous, Symbiotic, Intelligent, "
            "Sustainable) overall score is the weighted average of five "
            "dimension scores, each mapping to principles from Fath et al. "
            "(2019) regenerative economics framework."
        ),
        "interpret": (
            "**≥ 60 (Healthy)**: Organization shows strong ecosystemic health. "
            "**40–60 (Warning)**: Some dimensions need attention. "
            "**< 40 (Critical)**: Significant systemic issues across multiple "
            "dimensions."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    "oasis_open": {
        "label": "OPEN — Ability to Interconnect",
        "tooltip": (
            "Measures cross-boundary communication, resource exchange, "
            "and information flow capacity (Fath Principles 1, 3, 4)."
        ),
        "definition": (
            "The OPEN dimension assesses the organization's capacity to "
            "exchange resources and information across boundaries. It maps "
            "to Fath et al. Principles 1 (Cross-scale Circulation), "
            "3 (Reliable Inputs), and 4 (Healthy Outputs)."
        ),
        "interpret": (
            "**High (>70)**: Strong interconnectivity — information flows "
            "freely across boundaries. "
            "**Low (<30)**: Siloed departments, limited collaboration."
        ),
        "formula": "OPEN = 0.25 \\cdot C_n + 0.30 \\cdot H_{norm} + 0.25 \\cdot B_{avg} + 0.20 \\cdot CC",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    "oasis_autonomous": {
        "label": "AUTONOMOUS — Ability to Learn & Encode",
        "tooltip": (
            "Assesses learning capacity, knowledge retention, and "
            "self-reinforcing feedback loops (Fath Principles 2, 9)."
        ),
        "definition": (
            "The AUTONOMOUS dimension evaluates the organization's ability "
            "to learn from experience and encode routines through cycling "
            "and feedback. It maps to Fath et al. Principles 2 (Regenerative "
            "Re-investment) and 9 (Constructive vs Extractive)."
        ),
        "interpret": (
            "**High (>60)**: Strong learning capacity — feedback loops "
            "reinforce positive behaviors. "
            "**Low (<25)**: Weak knowledge encoding — lessons are not retained."
        ),
        "formula": "AUTO = 0.35 \\cdot FCI + 0.25 \\cdot Rec + 0.25 \\cdot \\frac{AMI}{H_{max}} + 0.15 \\cdot AC",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    "oasis_symbiotic": {
        "label": "SYMBIOTIC — Integration & Balance",
        "tooltip": (
            "Evaluates balanced cooperation and resource distribution "
            "between organizational elements (Fath Principles 5, 8)."
        ),
        "definition": (
            "The SYMBIOTIC dimension measures how well organizational "
            "elements cooperate and share resources. It maps to Fath et al. "
            "Principles 5 (Balance of Sizes) and 8 (Mutualism)."
        ),
        "interpret": (
            "**High (>70)**: Balanced resource distribution with mutual, "
            "reciprocal relationships. "
            "**Low (<35)**: Resource inequality and exploitative, one-way "
            "relationships."
        ),
        "formula": "SYM = 0.30 \\cdot (1 - Gini) + 0.25 \\cdot Q + 0.25 \\cdot \\frac{N_{eff}}{N} + 0.20 \\cdot M",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    "oasis_intelligent": {
        "label": "INTELLIGENT — Leverage Diverse Intelligence",
        "tooltip": (
            "Measures functional diversity, role specialization, and "
            "adaptive capacity (Fath Principles 7, 10)."
        ),
        "definition": (
            "The INTELLIGENT dimension assesses the organization's "
            "functional diversity and ability to leverage different types "
            "of expertise. It maps to Fath et al. Principles 7 (Sufficient "
            "Diversity) and 10 (Adaptive Learning)."
        ),
        "interpret": (
            "**High (>60)**: Rich role differentiation — diverse expertise "
            "is leveraged effectively. "
            "**Low (<30)**: Limited functional specialization."
        ),
        "formula": "INT = 0.35 \\cdot R_{norm} + 0.25 \\cdot D_f + 0.20 \\cdot \\frac{R}{N} + 0.20 \\cdot H_c",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    "oasis_sustainable": {
        "label": "SUSTAINABLE — Balance Order & Freedom",
        "tooltip": (
            "The central dimension — assesses the balance between efficiency "
            "and resilience in the Window of Vitality (Fath Principle 6)."
        ),
        "definition": (
            "The SUSTAINABLE dimension is the heart of the OASIS model. "
            "It evaluates whether the organization operates within the "
            "Window of Viability, balancing order (ascendency) with freedom "
            "(reserve). Maps to Fath et al. Principle 6."
        ),
        "interpret": (
            "**High (>70)**: Excellent efficiency-resilience balance, "
            "near-optimal α. "
            "**Low (<40)**: The system is either too rigid (α > 0.6) or "
            "too chaotic (α < 0.2)."
        ),
        "formula": "SUS = 0.30 \\cdot R_{norm} + 0.20 \\cdot W + 0.20 \\cdot RC_{norm} + 0.30 \\cdot \\alpha_{opt}",
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_OASIS,
    },

    # =====================================================================
    # REGENERATIVE ECONOMICS — 10 PRINCIPLES
    # =====================================================================
    "regen_in_out_balance": {
        "label": "Principle 1: In-Out Balance",
        "tooltip": (
            "Cross-scale circulation — healthy exchange of inputs and "
            "outputs across system boundaries."
        ),
        "definition": (
            "Fath et al. Principle 1: Systems must maintain robust "
            "cross-scale circulation. Resources flow in, are processed, "
            "and outputs flow out in a balanced manner."
        ),
        "interpret": (
            "**High**: Well-balanced exchange with environment. "
            "**Low**: Imbalanced — either hoarding or leaking resources."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Open",
        "category": CAT_REGEN,
    },

    "regen_sufficient_size": {
        "label": "Principle 2: Sufficient Size",
        "tooltip": (
            "Regenerative re-investment — the system reinvests enough "
            "to sustain and grow itself."
        ),
        "definition": (
            "Fath et al. Principle 2: Systems need sufficient internal "
            "re-investment to maintain and regenerate themselves. This is "
            "measured through cycling indices and feedback strength."
        ),
        "interpret": (
            "**High**: Strong internal reinvestment. "
            "**Low**: Extractive — not enough resources cycle back."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Autonomous",
        "category": CAT_REGEN,
    },

    "regen_hierarchy": {
        "label": "Principle 3: Hierarchy",
        "tooltip": (
            "Reliable inputs — appropriate hierarchical structure "
            "ensuring information flows through proper channels."
        ),
        "definition": (
            "Fath et al. Principle 3: Systems need appropriately layered "
            "hierarchy to ensure reliable processing of inputs and clear "
            "chains of information flow."
        ),
        "interpret": (
            "**High**: Clear, functional hierarchy. "
            "**Low**: Flat or confused structure."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Open",
        "category": CAT_REGEN,
    },

    "regen_material_basis": {
        "label": "Principle 4: Material Basis",
        "tooltip": (
            "Healthy outputs — the system produces meaningful, "
            "high-quality results from its inputs."
        ),
        "definition": (
            "Fath et al. Principle 4: Systems must transform inputs "
            "into meaningful outputs. This is reflected in the quality "
            "and throughput efficiency of the network."
        ),
        "interpret": (
            "**High**: Efficient transformation of inputs to outputs. "
            "**Low**: Wasteful processing."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Open",
        "category": CAT_REGEN,
    },

    "regen_mutuality": {
        "label": "Principle 5: Mutuality",
        "tooltip": (
            "Balance of sizes — equitable relationships where all "
            "participants benefit."
        ),
        "definition": (
            "Fath et al. Principle 5: Healthy systems exhibit balanced "
            "reciprocal relationships. Resources flow bidirectionally, "
            "and no single actor extracts disproportionately."
        ),
        "interpret": (
            "**High**: Reciprocal, mutualistic relationships. "
            "**Low**: One-sided, exploitative dynamics."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Symbiotic",
        "category": CAT_REGEN,
    },

    "regen_diversity": {
        "label": "Principle 6: Diversity",
        "tooltip": (
            "Sufficient diversity of roles and functions to ensure "
            "adaptive capacity."
        ),
        "definition": (
            "Fath et al. Principle 7 (mapped here as diversity): Systems "
            "need sufficient functional diversity. The number of distinct "
            "roles and their distribution affects adaptive capacity."
        ),
        "interpret": (
            "**High**: Rich diversity of expertise and functions. "
            "**Low**: Monoculture — limited functional types."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Intelligent",
        "category": CAT_REGEN,
    },

    "regen_circulation": {
        "label": "Principle 7: Circulation",
        "tooltip": (
            "Internal cycling of resources, information, and value "
            "through the organization."
        ),
        "definition": (
            "Fath et al. Principle 2/9: Resources must circulate within "
            "the system rather than flowing only in one direction. Measured "
            "by the Finn Cycling Index."
        ),
        "interpret": (
            "**High**: Active internal circulation. "
            "**Low**: Linear, one-pass flow."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Autonomous",
        "category": CAT_REGEN,
    },

    "regen_reserve_capacity": {
        "label": "Principle 8: Reserve Capacity",
        "tooltip": (
            "Maintaining sufficient reserves and redundancy to handle "
            "unexpected disruptions."
        ),
        "definition": (
            "Fath et al. Principle 6: Systems need reserve capacity — "
            "redundant pathways and unused potential — to absorb shocks. "
            "Measured through overhead ratio and redundancy."
        ),
        "interpret": (
            "**High**: Well-buffered against disruptions. "
            "**Low**: Operating at full capacity with no margin."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Sustainable",
        "category": CAT_REGEN,
    },

    "regen_efficiency": {
        "label": "Principle 9: Efficiency",
        "tooltip": (
            "Constructive efficiency — not just lean operations but "
            "purposeful, value-creating throughput."
        ),
        "definition": (
            "Fath et al. Principle 9: Efficiency should be constructive, "
            "not extractive. The system should maximize value creation "
            "through its organized pathways without sacrificing resilience."
        ),
        "interpret": (
            "**High**: Purposeful, constructive efficiency. "
            "**Low**: Wasteful or extractive operations."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Autonomous",
        "category": CAT_REGEN,
    },

    "regen_balance": {
        "label": "Principle 10: Balance",
        "tooltip": (
            "The overarching principle — balance between efficiency "
            "and resilience, order and freedom."
        ),
        "definition": (
            "Fath et al. Principle 6/10: The master principle. All "
            "healthy systems balance efficiency (ascendency) with "
            "resilience (overhead). This is captured by the Window of "
            "Viability and Robustness metrics."
        ),
        "interpret": (
            "**High**: Excellent balance between competing demands. "
            "**Low**: Imbalanced — tilting toward rigidity or chaos."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Sustainable",
        "category": CAT_REGEN,
    },

    # =====================================================================
    # NETWORK STRUCTURE & TOPOLOGY
    # =====================================================================
    "nodes": {
        "label": "Nodes",
        "tooltip": (
            "Number of distinct entities (departments, teams, actors) "
            "in the network."
        ),
        "definition": (
            "The count of vertices in the directed graph. Each node "
            "represents an organizational unit, department, species, "
            "or actor that participates in resource flows."
        ),
        "interpret": (
            "**High**: Large, complex organization with many actors. "
            "**Low**: Small, simple network."
        ),
        "category": CAT_NETWORK,
    },

    "edges": {
        "label": "Edges",
        "tooltip": (
            "Number of active connections (resource/information flows) "
            "between nodes."
        ),
        "definition": (
            "The count of directed links with non-zero flow. Each edge "
            "represents a resource, information, or value transfer between "
            "two nodes."
        ),
        "interpret": (
            "**High**: Dense connectivity — many relationships. "
            "**Low**: Sparse — few active connections."
        ),
        "category": CAT_NETWORK,
    },

    "network_density": {
        "label": "Network Density",
        "tooltip": (
            "Fraction of all possible connections that actually exist — "
            "how interconnected the network is."
        ),
        "definition": (
            "Network Density is the ratio of actual edges to the maximum "
            "possible edges in a directed graph: ρ = L / (N × (N-1)). "
            "Ranges from 0 (no connections) to 1 (fully connected)."
        ),
        "interpret": (
            "**High (>0.5)**: Highly interconnected — everyone talks "
            "to everyone. "
            "**Low (<0.1)**: Sparse — isolated clusters likely."
        ),
        "formula": "\\rho = \\frac{L}{N(N-1)}",
        "category": CAT_NETWORK,
    },

    "connectance": {
        "label": "Connectance",
        "tooltip": (
            "Ecological connectivity measure — fraction of possible "
            "connections realized in the food web / flow network."
        ),
        "definition": (
            "Connectance is the fraction of possible directed connections "
            "that carry non-zero flow. In ecology, it measures the "
            "completeness of the food web."
        ),
        "interpret": (
            "**High**: Well-connected network. "
            "**Low**: Few connections relative to potential."
        ),
        "formula": "C_n = \\frac{L}{N(N-1)}",
        "oasis_map": "Open",
        "category": CAT_NETWORK,
    },

    "avg_path_length": {
        "label": "Average Path Length",
        "tooltip": (
            "Mean number of steps between any two nodes — how quickly "
            "information or resources can travel across the network."
        ),
        "definition": (
            "Average shortest path length between all pairs of reachable "
            "nodes. Indicates the 'degrees of separation' in the network."
        ),
        "interpret": (
            "**Low (<3)**: Efficient communication, small-world property. "
            "**High (>5)**: Information must traverse many intermediaries."
        ),
        "formula": "\\langle l \\rangle = \\frac{1}{N(N-1)} \\sum_{i \\neq j} d(i,j)",
        "category": CAT_NETWORK,
    },

    "clustering_coefficient": {
        "label": "Clustering Coefficient",
        "tooltip": (
            "Probability that two connected neighbors of a node are also "
            "connected — measures local group cohesion."
        ),
        "definition": (
            "The average clustering coefficient measures the tendency of "
            "nodes to form tightly knit groups. High clustering indicates "
            "modular, team-like structures."
        ),
        "interpret": (
            "**High (>0.5)**: Strong local clusters — teams work closely "
            "together. "
            "**Low (<0.1)**: Weak local structure — connections are diffuse."
        ),
        "citation": REF_WATTS_STROGATZ_1998,
        "oasis_map": "Open",
        "category": CAT_NETWORK,
    },

    "degree_centralization": {
        "label": "Degree Centralization",
        "tooltip": (
            "How concentrated connections are around a few hub nodes — "
            "a star network has centralization = 1."
        ),
        "definition": (
            "Degree Centralization measures the extent to which the "
            "network is dominated by a single node or small group. "
            "Calculated as the sum of differences from the maximum "
            "degree, normalized by the theoretical maximum."
        ),
        "interpret": (
            "**High (>0.7)**: Hub-and-spoke — a few nodes control most "
            "connections. "
            "**Low (<0.3)**: Distributed — connections are spread evenly."
        ),
        "category": CAT_NETWORK,
    },

    "link_density": {
        "label": "Link Density",
        "tooltip": (
            "Average number of connections per node — how many "
            "relationships each entity maintains."
        ),
        "definition": (
            "Link Density is the ratio of total edges to total nodes "
            "(L/N). It indicates the average 'workload' of connections "
            "per organizational unit."
        ),
        "interpret": (
            "**High**: Nodes maintain many relationships. "
            "**Low**: Nodes have few connections."
        ),
        "formula": "LD = L / N",
        "category": CAT_NETWORK,
    },

    # =====================================================================
    # FLOW STATISTICS
    # =====================================================================
    "total_flow": {
        "label": "Total Flow",
        "tooltip": "Sum of all non-zero flows in the network (same as TST).",
        "definition": (
            "The aggregate of all flow values in the network matrix. "
            "Equivalent to Total System Throughput (TST)."
        ),
        "interpret": (
            "**High**: Large-scale resource movement. "
            "**Low**: Limited flow activity."
        ),
        "category": CAT_FLOW,
    },

    "active_connections": {
        "label": "Active Connections",
        "tooltip": "Count of non-zero flows in the matrix — edges that carry resources.",
        "definition": (
            "The number of matrix cells with T_ij > 0. Represents the "
            "actual links carrying flow, as opposed to potential links."
        ),
        "interpret": (
            "**High**: Many pathways are in use. "
            "**Low**: Most potential connections are dormant."
        ),
        "category": CAT_FLOW,
    },

    "avg_flow": {
        "label": "Average Flow",
        "tooltip": "Mean flow value across all active connections.",
        "definition": "The arithmetic mean of all non-zero flow values.",
        "interpret": (
            "Context-dependent — compare with median to detect skew. "
            "If avg >> median, a few large flows dominate."
        ),
        "category": CAT_FLOW,
    },

    "median_flow": {
        "label": "Median Flow",
        "tooltip": "Middle value of all active flows — robust to outliers.",
        "definition": (
            "The median of non-zero flow values. Less sensitive to "
            "extreme outliers than the mean."
        ),
        "interpret": (
            "**If median << mean**: Flows are highly skewed — a few "
            "dominant pathways. "
            "**If median ≈ mean**: Flows are relatively even."
        ),
        "category": CAT_FLOW,
    },

    "max_flow": {
        "label": "Maximum Flow",
        "tooltip": "Largest single flow in the network — the dominant connection.",
        "definition": "The maximum value among all flow matrix entries.",
        "interpret": "Identifies the single strongest relationship in the network.",
        "category": CAT_FLOW,
    },

    "min_flow": {
        "label": "Minimum Flow (>0)",
        "tooltip": "Smallest non-zero flow — the weakest active connection.",
        "definition": "The minimum positive value in the flow matrix.",
        "interpret": (
            "Large gap between min and max indicates high flow heterogeneity."
        ),
        "category": CAT_FLOW,
    },

    "flow_std_dev": {
        "label": "Flow Standard Deviation",
        "tooltip": "Spread of flow values — how variable the connection strengths are.",
        "definition": "Standard deviation of all non-zero flow values.",
        "interpret": (
            "**High**: Wide variation in connection strengths. "
            "**Low**: Relatively uniform flows."
        ),
        "category": CAT_FLOW,
    },

    "coeff_variation": {
        "label": "Coefficient of Variation",
        "tooltip": (
            "Relative variability of flows (σ/μ) — dimensionless measure "
            "of flow heterogeneity."
        ),
        "definition": (
            "Ratio of standard deviation to mean of flow values. "
            "Allows comparison of variability across networks of "
            "different scales."
        ),
        "interpret": (
            "**CV > 1**: Highly heterogeneous flows. "
            "**CV < 0.5**: Relatively homogeneous flows."
        ),
        "formula": "CV = \\sigma / \\mu",
        "category": CAT_FLOW,
    },

    # =====================================================================
    # ROLES & FUNCTIONAL SPECIALIZATION
    # =====================================================================
    "num_roles": {
        "label": "Number of Roles (R)",
        "tooltip": (
            "Number of functionally distinct roles in the network — "
            "derived as exp(AMI)."
        ),
        "definition": (
            "The effective number of functional roles is the exponential "
            "of Average Mutual Information: R = exp(AMI). It quantifies "
            "functional differentiation in the network."
        ),
        "interpret": (
            "**High**: Many specialized functions — diverse expertise. "
            "**Low**: Few distinct roles — limited specialization."
        ),
        "formula": "R = e^{AMI}",
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "oasis_map": "Intelligent",
        "category": CAT_ROLES,
    },

    "effective_nodes": {
        "label": "Effective Nodes (N_eff)",
        "tooltip": (
            "Flow-weighted number of active nodes — how many nodes "
            "carry significant throughput."
        ),
        "definition": (
            "Effective Nodes is a flow-weighted count that discounts "
            "nodes with negligible throughput. It tells you how many "
            "nodes are truly active participants in the network."
        ),
        "interpret": (
            "**N_eff ≈ N**: All nodes are active participants. "
            "**N_eff << N**: Many nodes are peripheral or inactive."
        ),
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "oasis_map": "Symbiotic",
        "category": CAT_ROLES,
    },

    "effective_flows": {
        "label": "Effective Flows (F_eff)",
        "tooltip": (
            "Exponential of flow diversity — the effective number of "
            "distinct flow pathways."
        ),
        "definition": (
            "Effective Flows = exp(H), where H is Flow Diversity. "
            "It measures how many truly independent pathways carry "
            "significant flow."
        ),
        "interpret": (
            "**High**: Many independent flow pathways. "
            "**Low**: Flow concentrated in a few paths."
        ),
        "formula": "F_{eff} = e^{H}",
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "category": CAT_ROLES,
    },

    "effective_connectivity": {
        "label": "Effective Connectivity (C_eff)",
        "tooltip": (
            "Effective flows per effective node — how many meaningful "
            "connections each active node maintains."
        ),
        "definition": (
            "Effective Connectivity = F_eff / N_eff. It measures the "
            "average number of meaningful connections per active node."
        ),
        "interpret": (
            "**High**: Dense effective connections per node. "
            "**Low**: Sparse meaningful connectivity."
        ),
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "category": CAT_ROLES,
    },

    "roles_per_node": {
        "label": "Roles per Node",
        "tooltip": (
            "Average number of functional roles each effective node "
            "fulfills — R / N_eff."
        ),
        "definition": (
            "The ratio of Number of Roles to Effective Nodes. Indicates "
            "whether nodes specialize in single functions or wear multiple "
            "hats."
        ),
        "interpret": (
            "**High (>2)**: Nodes fill multiple roles — versatile but "
            "potentially overburdened. "
            "**Low (~1)**: Clear single-role assignment."
        ),
        "formula": "RPH = R / N_{eff}",
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "oasis_map": "Intelligent",
        "category": CAT_ROLES,
    },

    "specialization_index": {
        "label": "Specialization Index",
        "tooltip": (
            "Number of roles relative to actual nodes — R / N."
        ),
        "definition": (
            "Specialization Index = R / N_actual. It tells you how "
            "functionally specialized the network is relative to its "
            "actual size."
        ),
        "interpret": (
            "**> 1**: More roles than nodes — high functional diversity. "
            "**< 1**: Fewer roles than nodes — under-specialized."
        ),
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "category": CAT_ROLES,
    },

    "node_utilization": {
        "label": "Node Utilization",
        "tooltip": (
            "Fraction of nodes that are effectively active — "
            "N_eff / N_actual."
        ),
        "definition": (
            "Node Utilization = N_eff / N_actual. Measures what "
            "proportion of the organization is actively contributing "
            "to throughput."
        ),
        "interpret": (
            "**High (>0.8)**: Most nodes are active participants. "
            "**Low (<0.5)**: Many nodes are underutilized."
        ),
        "oasis_map": "Symbiotic",
        "category": CAT_ROLES,
    },

    # =====================================================================
    # CENTRALITY MEASURES
    # =====================================================================
    "degree_centrality": {
        "label": "Degree Centrality",
        "tooltip": (
            "How many connections a node has — identifies the most "
            "connected entities."
        ),
        "definition": (
            "Degree Centrality is the fraction of other nodes a given "
            "node is connected to. For directed graphs, considers both "
            "in-degree and out-degree."
        ),
        "interpret": (
            "**High**: Hub node — central to many relationships. "
            "**Low**: Peripheral node with few connections."
        ),
        "category": CAT_CENTRALITY,
    },

    "betweenness_centrality": {
        "label": "Betweenness Centrality",
        "tooltip": (
            "How often a node lies on shortest paths between others — "
            "identifies bridges and bottlenecks."
        ),
        "definition": (
            "Betweenness Centrality measures the fraction of all "
            "shortest paths in the network that pass through a node. "
            "High-betweenness nodes are critical intermediaries."
        ),
        "interpret": (
            "**High**: Bridge/broker — removing this node would "
            "disrupt many connections. "
            "**Low**: Not a critical intermediary."
        ),
        "oasis_map": "Open",
        "category": CAT_CENTRALITY,
    },

    "pagerank": {
        "label": "PageRank",
        "tooltip": (
            "Node importance based on incoming connections from other "
            "important nodes — Google's algorithm applied to org networks."
        ),
        "definition": (
            "PageRank computes the importance of each node based on "
            "the importance of nodes linking to it, with a damping "
            "factor of 0.85. Originally developed for web page ranking."
        ),
        "interpret": (
            "**High**: Influential node that receives resources from "
            "other influential nodes. "
            "**Low**: Peripheral or receiving only from minor sources."
        ),
        "category": CAT_CENTRALITY,
    },

    "closeness_centrality": {
        "label": "Closeness Centrality",
        "tooltip": (
            "How close a node is to all others — identifies nodes that "
            "can reach the entire network quickly."
        ),
        "definition": (
            "Closeness Centrality is the reciprocal of the average "
            "shortest path distance from a node to all other nodes. "
            "High closeness means quick access to the whole network."
        ),
        "interpret": (
            "**High**: Can quickly reach all parts of the network. "
            "**Low**: Far from most other nodes."
        ),
        "category": CAT_CENTRALITY,
    },

    # =====================================================================
    # COMMUNITY STRUCTURE
    # =====================================================================
    "communities": {
        "label": "Communities",
        "tooltip": (
            "Number of distinct groups/clusters detected in the network "
            "using community detection algorithms."
        ),
        "definition": (
            "The number of distinct communities identified by the Louvain "
            "algorithm. Communities are densely connected subgroups with "
            "fewer connections between groups."
        ),
        "interpret": (
            "**Many communities**: Modular organization with distinct "
            "divisions. "
            "**Few/one community**: Monolithic or highly integrated."
        ),
        "category": CAT_COMMUNITY,
    },

    "modularity": {
        "label": "Modularity (Q)",
        "tooltip": (
            "Strength of community structure — higher means clearer "
            "separation into distinct groups."
        ),
        "definition": (
            "Modularity Q measures how much the network decomposes into "
            "distinct communities compared to a random network. Ranges "
            "from -0.5 to 1.0."
        ),
        "interpret": (
            "**Q > 0.3**: Strong community structure — clear divisions. "
            "**Q < 0.1**: Weak community structure — no obvious groupings."
        ),
        "oasis_map": "Symbiotic",
        "category": CAT_COMMUNITY,
    },

    "degree_assortativity": {
        "label": "Degree Assortativity",
        "tooltip": (
            "Whether high-degree nodes connect to other high-degree nodes "
            "(assortative) or to low-degree nodes (disassortative)."
        ),
        "definition": (
            "Degree Assortativity measures the correlation between the "
            "degrees of connected node pairs. Positive values indicate "
            "hubs connect to hubs; negative values indicate hubs connect "
            "to peripheral nodes."
        ),
        "interpret": (
            "**Positive**: Assortative — hubs cluster together (typical "
            "of social networks). "
            "**Negative**: Disassortative — hub-and-spoke pattern (typical "
            "of biological networks)."
        ),
        "category": CAT_COMMUNITY,
    },

    "rich_club": {
        "label": "Rich Club Coefficient",
        "tooltip": (
            "Whether the most connected nodes preferentially connect to "
            "each other — 'rich get richer' effect."
        ),
        "definition": (
            "The Rich Club Coefficient measures the tendency of "
            "high-degree nodes to form tightly interconnected cliques. "
            "Computed as the fraction of possible edges among top-k "
            "nodes that actually exist."
        ),
        "interpret": (
            "**High**: Elite cluster — top nodes form an inner circle. "
            "**Low**: Hub nodes do not preferentially connect."
        ),
        "category": CAT_COMMUNITY,
    },

    # =====================================================================
    # ROBUSTNESS & RESILIENCE
    # =====================================================================
    "random_failure_robustness": {
        "label": "Random Failure Robustness",
        "tooltip": (
            "How well the network holds together when nodes fail "
            "randomly — like random employee absences."
        ),
        "definition": (
            "Measures the average size of the largest connected component "
            "as random nodes are progressively removed. Higher values "
            "indicate the network degrades gracefully under random failures."
        ),
        "interpret": (
            "**High (>0.7)**: Robust to random disruptions. "
            "**Low (<0.3)**: Fragile — random failures quickly fragment "
            "the network."
        ),
        "category": CAT_ROBUST,
    },

    "targeted_attack_robustness": {
        "label": "Targeted Attack Robustness",
        "tooltip": (
            "How well the network survives when the most-connected "
            "nodes are removed first — like losing key leaders."
        ),
        "definition": (
            "Measures the average size of the largest connected component "
            "as the highest-degree nodes are progressively removed. "
            "Tests vulnerability to strategic disruption."
        ),
        "interpret": (
            "**High (>0.5)**: Resilient even when key nodes are lost. "
            "**Low (<0.3)**: Highly dependent on hub nodes — vulnerable "
            "to targeted disruption."
        ),
        "category": CAT_ROBUST,
    },

    "percolation_threshold": {
        "label": "Percolation Threshold",
        "tooltip": (
            "Critical fraction of nodes that must fail before the "
            "network fragments — higher is more resilient."
        ),
        "definition": (
            "The estimated fraction of nodes that can be removed "
            "before the network loses its giant component. Approximated "
            "as 1 / average_degree."
        ),
        "interpret": (
            "**High (>0.3)**: Network tolerates significant node loss. "
            "**Low (<0.1)**: Very fragile — few failures cause fragmentation."
        ),
        "formula": "p_c \\approx 1 / \\langle k \\rangle",
        "category": CAT_ROBUST,
    },

    "path_redundancy": {
        "label": "Path Redundancy",
        "tooltip": (
            "Average number of alternative routes between nodes — "
            "more alternatives means better resilience."
        ),
        "definition": (
            "The average number of node-independent paths between "
            "sampled node pairs. Higher values indicate more backup "
            "routes available when primary paths fail."
        ),
        "interpret": (
            "**High (>3)**: Rich alternative routing. "
            "**Low (≈1)**: Single points of failure on most routes."
        ),
        "category": CAT_ROBUST,
    },

    # =====================================================================
    # FLOW CHARACTERISTICS (NETWORK ANALYSIS)
    # =====================================================================
    "flow_gini": {
        "label": "Flow Gini Coefficient",
        "tooltip": (
            "Inequality in flow distribution — 0 means perfectly equal, "
            "1 means all flow in one connection."
        ),
        "definition": (
            "The Gini coefficient applied to flow values. Measures "
            "how unequally resources are distributed across connections."
        ),
        "interpret": (
            "**High (>0.7)**: Extreme inequality — a few connections "
            "carry almost all flow. "
            "**Low (<0.3)**: Relatively egalitarian distribution."
        ),
        "oasis_map": "Symbiotic",
        "category": CAT_FLOW,
    },

    "flow_heterogeneity": {
        "label": "Flow Heterogeneity",
        "tooltip": (
            "Variability in flow strengths across connections — "
            "coefficient of variation of flows."
        ),
        "definition": (
            "Coefficient of variation of flow values: std(flows) / "
            "mean(flows). Measures how variable connection strengths are."
        ),
        "interpret": (
            "**High (>2)**: Extremely variable flow strengths. "
            "**Low (<0.5)**: Relatively uniform flows."
        ),
        "category": CAT_FLOW,
    },

    "throughput_efficiency": {
        "label": "Throughput Efficiency",
        "tooltip": (
            "Actual flow as a fraction of maximum possible flow — "
            "how fully the network's capacity is utilized."
        ),
        "definition": (
            "Ratio of actual total flow to the theoretical maximum "
            "(if all connections carried the maximum observed flow). "
            "Measures capacity utilization."
        ),
        "interpret": (
            "**High (>0.5)**: Network heavily utilized. "
            "**Low (<0.1)**: Most capacity unused."
        ),
        "category": CAT_FLOW,
    },

    "flow_reciprocity": {
        "label": "Flow Reciprocity",
        "tooltip": (
            "Proportion of connections that are bidirectional — "
            "mutual exchange vs. one-way flows."
        ),
        "definition": (
            "The fraction of edges that have a reciprocal edge "
            "(if A→B exists, does B→A also exist?). Measures the "
            "mutualism in the network."
        ),
        "interpret": (
            "**High (>0.5)**: Many mutual, reciprocal relationships. "
            "**Low (<0.2)**: Predominantly one-way flows."
        ),
        "oasis_map": "Autonomous, Symbiotic",
        "category": CAT_FLOW,
    },

    # =====================================================================
    # BALANCE INDICATORS
    # =====================================================================
    "organization_ratio": {
        "label": "Organization (A/C Ratio)",
        "tooltip": (
            "Fraction of capacity used for organization — same as α. "
            "Optimal range 0.35–0.40."
        ),
        "definition": (
            "The Ascendency/Capacity ratio indicates how much of the "
            "system's potential is organized. Equivalent to α."
        ),
        "interpret": (
            "**< 0.2**: Too Chaotic — needs more structure. "
            "**0.35–0.40**: Optimal balance. "
            "**> 0.6**: Too Rigid — needs more flexibility."
        ),
        "category": CAT_BALANCE,
    },

    "flexibility_ratio": {
        "label": "Flexibility (Φ/C Ratio)",
        "tooltip": (
            "Fraction of capacity kept as reserve — the complement of "
            "organization ratio."
        ),
        "definition": (
            "The Overhead/Capacity ratio measures how much system "
            "potential is maintained as flexible reserve."
        ),
        "interpret": (
            "**< 0.4**: Low Reserve — operating too lean. "
            "**0.4–0.8**: Good Balance. "
            "**> 0.8**: High Redundancy — excess waste."
        ),
        "category": CAT_BALANCE,
    },

    "eff_red_balance": {
        "label": "Efficiency/Redundancy Balance",
        "tooltip": (
            "Ratio of organized activity to reserve capacity — "
            "A/Φ indicates the tilt between efficiency and resilience."
        ),
        "definition": (
            "The ratio of Ascendency to Overhead (A/Φ). Values near 1 "
            "indicate balance; much higher means efficiency-heavy; much "
            "lower means redundancy-heavy."
        ),
        "interpret": (
            "**>> 1**: Tilted toward efficiency — organized but fragile. "
            "**<< 1**: Tilted toward resilience — resilient but inefficient. "
            "**≈ 0.6**: Near optimal (corresponds to α ≈ 0.37)."
        ),
        "category": CAT_BALANCE,
    },

    # =====================================================================
    # HEALTH ASSESSMENT LABELS
    # =====================================================================
    "status_optimal": {
        "label": "OPTIMAL Status",
        "tooltip": (
            "System at peak sustainability — α near 0.37 with maximum "
            "robustness."
        ),
        "definition": (
            "The system's Relative Ascendency (α) is in the optimal "
            "range of 0.35–0.40, where Robustness R = -α·ln(α) is "
            "maximized. The organization achieves the best possible "
            "balance between efficiency and resilience."
        ),
        "interpret": "Maintain current balance — this is the target zone.",
        "category": CAT_HEALTH,
    },

    "status_viable": {
        "label": "VIABLE Status",
        "tooltip": (
            "System is within the Window of Viability (0.2 ≤ α ≤ 0.6) "
            "but not at the optimal point."
        ),
        "definition": (
            "The system operates within sustainable bounds but has room "
            "for improvement toward the optimal α ≈ 0.37."
        ),
        "interpret": (
            "Functional and sustainable, but could improve. Move α "
            "toward 0.37 for maximum robustness."
        ),
        "category": CAT_HEALTH,
    },

    "status_unsustainable": {
        "label": "UNSUSTAINABLE Status",
        "tooltip": (
            "System outside the Window of Viability — either too rigid "
            "(α > 0.6) or too chaotic (α < 0.2)."
        ),
        "definition": (
            "The system's efficiency-resilience balance has crossed "
            "sustainable thresholds. If α > 0.6 the organization is "
            "over-optimized and brittle. If α < 0.2 it lacks sufficient "
            "structure."
        ),
        "interpret": (
            "**α > 0.6**: Reduce rigidity — add redundant pathways, "
            "diversify connections. "
            "**α < 0.2**: Increase organization — consolidate flows, "
            "strengthen key pathways."
        ),
        "category": CAT_HEALTH,
    },

    # =====================================================================
    # SMALL WORLD METRICS
    # =====================================================================
    "small_world_sigma": {
        "label": "Small World Sigma (σ)",
        "tooltip": (
            "Ratio comparing clustering and path length to random "
            "networks — σ > 1 indicates small-world property."
        ),
        "definition": (
            "Small World Sigma = (C_actual/C_random) / (L_actual/L_random). "
            "A network is small-world if it has high clustering like a "
            "lattice but short paths like a random graph."
        ),
        "interpret": (
            "**σ > 1**: Small-world network — efficient information flow "
            "with strong local clustering. "
            "**σ ≈ 1**: Random network."
        ),
        "citation": REF_WATTS_STROGATZ_1998,
        "category": CAT_NETWORK,
    },

    # =====================================================================
    # NETWORK HEALTH SUMMARY
    # =====================================================================
    "network_health_connectivity": {
        "label": "Connectivity Health",
        "tooltip": "Assessment of how well-connected the network is overall.",
        "definition": (
            "Composite score based on density, average path length, "
            "and component structure. A well-connected network enables "
            "efficient resource flow."
        ),
        "interpret": (
            "**Good**: Network is well-connected. "
            "**Poor**: Isolated components or excessive fragmentation."
        ),
        "category": CAT_HEALTH,
    },

    "network_health_modularity": {
        "label": "Modularity Health",
        "tooltip": "Whether the community structure supports functional organization.",
        "definition": (
            "Assessment of whether the network has appropriate modular "
            "structure — neither too monolithic nor too fragmented."
        ),
        "interpret": (
            "**Good**: Clear, functional divisions without over-fragmentation. "
            "**Poor**: Either no structure or excessive siloing."
        ),
        "category": CAT_HEALTH,
    },

    "network_health_robustness": {
        "label": "Robustness Health",
        "tooltip": "How well the network can withstand node failures and attacks.",
        "definition": (
            "Composite score based on random failure robustness, "
            "targeted attack robustness, and path redundancy."
        ),
        "interpret": (
            "**Good**: Network degrades gracefully under stress. "
            "**Poor**: Fragile — vulnerable to disruption."
        ),
        "category": CAT_HEALTH,
    },

    "network_health_efficiency_score": {
        "label": "Efficiency Health",
        "tooltip": "How efficiently the network utilizes its connections for flow.",
        "definition": (
            "Assessment of throughput efficiency and flow distribution "
            "characteristics."
        ),
        "interpret": (
            "**Good**: Efficient flow patterns. "
            "**Poor**: Wasteful or under-utilized connections."
        ),
        "category": CAT_HEALTH,
    },

    # =====================================================================
    # VISUALIZATIONS
    # =====================================================================
    "viz_sankey": {
        "label": "Directed Network Flow Diagram (Sankey)",
        "tooltip": (
            "Visual representation of all flows between nodes — wider "
            "bands indicate larger flows."
        ),
        "definition": (
            "A Sankey diagram displays the magnitude of flows between "
            "nodes as proportional band widths. It provides an intuitive "
            "view of where resources, information, or value move in the "
            "network."
        ),
        "interpret": (
            "Look for: dominant flows (widest bands), bottleneck nodes "
            "(many flows converging), isolated nodes (few/no bands), "
            "and balance (even vs. skewed distribution)."
        ),
        "category": CAT_VIZ,
    },

    "viz_heatmap": {
        "label": "Network Flow Heatmap",
        "tooltip": (
            "Color-coded matrix showing flow intensity between every "
            "pair of nodes — dark = strong flow."
        ),
        "definition": (
            "A heatmap of the flow matrix where each cell (i,j) shows "
            "the flow from node i to node j. Color intensity represents "
            "flow magnitude."
        ),
        "interpret": (
            "**Diagonal clusters**: Indicate modular structure. "
            "**Hot rows/columns**: Hub nodes. "
            "**Symmetric patterns**: Reciprocal relationships."
        ),
        "category": CAT_VIZ,
    },

    "viz_network_spring": {
        "label": "Network Diagram — Spring Layout",
        "tooltip": (
            "Force-directed graph where connected nodes pull together "
            "and disconnected nodes push apart."
        ),
        "definition": (
            "A spring-layout (Fruchterman-Reingold) visualization where "
            "node positions are computed by simulating attractive forces "
            "along edges and repulsive forces between all nodes. Node "
            "size reflects throughput; color reflects total flow."
        ),
        "interpret": (
            "**Clusters**: Groups of tightly connected nodes. "
            "**Central nodes**: Highly connected hubs. "
            "**Peripheral nodes**: Weakly connected participants."
        ),
        "category": CAT_VIZ,
    },

    "viz_network_directed": {
        "label": "Network Diagram — Directed View",
        "tooltip": (
            "Network graph with arrows showing flow direction and "
            "curved edges for clarity."
        ),
        "definition": (
            "A directed graph visualization with arrows indicating flow "
            "direction. Edge thickness reflects flow magnitude. Uses "
            "curved Bezier edges to distinguish bidirectional connections."
        ),
        "interpret": (
            "**Arrow direction**: Shows resource/information flow direction. "
            "**Edge thickness**: Thicker = stronger flow. "
            "**Bidirectional arrows**: Mutual, reciprocal relationships."
        ),
        "category": CAT_VIZ,
    },

    "viz_window_of_viability": {
        "label": "Window of Viability Plot",
        "tooltip": (
            "Shows where the organization sits on the efficiency-resilience "
            "spectrum relative to the viable zone."
        ),
        "definition": (
            "A plot showing the Robustness function R = -α·ln(α) with "
            "the organization's current position marked. The green zone "
            "(α = 0.2–0.6) is the Window of Viability; the peak at "
            "α ≈ 0.368 is the theoretical optimum."
        ),
        "interpret": (
            "**Dot in green zone**: System is viable. "
            "**Dot at peak**: Optimal balance achieved. "
            "**Dot outside green zone**: Action needed to rebalance."
        ),
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_VIZ,
    },

    "viz_radar": {
        "label": "System Health Radar Chart",
        "tooltip": (
            "Multi-dimensional view of key health indicators — a balanced "
            "system shows an even, circular profile."
        ),
        "definition": (
            "A radar (spider) chart displaying multiple health metrics "
            "simultaneously. Each axis represents a different metric "
            "normalized to a common scale."
        ),
        "interpret": (
            "**Round shape**: Balanced system with even performance. "
            "**Spiky shape**: Uneven — strong in some areas, weak in "
            "others. Focus improvement on the shortest spokes."
        ),
        "category": CAT_VIZ,
    },

    "viz_oasis_radar": {
        "label": "OASIS Health Profile (Radar)",
        "tooltip": (
            "Radar chart of the five OASIS dimensions — shows the "
            "organization's ecosystemic health profile at a glance."
        ),
        "definition": (
            "A radar chart displaying the five OASIS dimension scores "
            "(Open, Autonomous, Symbiotic, Intelligent, Sustainable) "
            "on a 0–100 scale. A balanced, healthy organization shows "
            "an even, outward-reaching profile."
        ),
        "interpret": (
            "**Even profile**: Balanced ecosystemic health. "
            "**Collapsed dimension**: Critical weakness — focus "
            "improvement there. "
            "**All high**: Excellent overall health."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "category": CAT_VIZ,
    },

    "viz_robustness_efficiency": {
        "label": "System Robustness vs Network Efficiency",
        "tooltip": (
            "Scatter plot showing the organization's position relative "
            "to the theoretical optimum of efficiency vs. robustness."
        ),
        "definition": (
            "A plot with Network Efficiency (α) on the x-axis and "
            "Robustness (R) on the y-axis. The theoretical curve "
            "R = -α·ln(α) defines the maximum achievable robustness "
            "for each efficiency level."
        ),
        "interpret": (
            "**Near the peak**: Optimal balance. "
            "**Far left**: Too chaotic. "
            "**Far right**: Too rigid."
        ),
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_VIZ,
    },

    "viz_degree_distribution": {
        "label": "Degree Distribution",
        "tooltip": (
            "Histogram of how many connections each node has — reveals "
            "whether the network is hub-dominated or egalitarian."
        ),
        "definition": (
            "Histograms of in-degree and out-degree distributions. "
            "The shape reveals the network's topology: power-law "
            "(scale-free), Poisson (random), or uniform (regular)."
        ),
        "interpret": (
            "**Long tail**: Scale-free — few hubs, many peripheral nodes. "
            "**Bell curve**: Random network. "
            "**Flat**: Regular — all nodes have similar connectivity."
        ),
        "category": CAT_VIZ,
    },

    "viz_actual_vs_effective": {
        "label": "Actual vs Effective Network Components",
        "tooltip": (
            "Bar chart comparing actual node/edge counts to their "
            "effective (flow-weighted) equivalents."
        ),
        "definition": (
            "A grouped bar chart comparing actual network components "
            "(nodes, flows, connectivity) to their effective "
            "(information-theoretic) counterparts from Zorach & "
            "Ulanowicz (2003)."
        ),
        "interpret": (
            "**Effective ≈ Actual**: All components are actively "
            "contributing. "
            "**Effective << Actual**: Many components are under-utilized."
        ),
        "citation": REF_ZORACH_2003,
        "doi": DOI_ZORACH_2003,
        "category": CAT_VIZ,
    },

    # =====================================================================
    # MATH VERIFICATION
    # =====================================================================
    "math_check_capacity": {
        "label": "C = A + Φ Check",
        "tooltip": (
            "Verifies the fundamental relationship: Development Capacity "
            "must equal Ascendency plus Overhead."
        ),
        "definition": (
            "A mathematical consistency check ensuring that C = A + Φ "
            "holds within numerical precision. Deviations indicate a "
            "calculation error."
        ),
        "interpret": (
            "**✅ Valid**: Calculations are internally consistent. "
            "**⚠️ Error > 0.01**: Numerical precision issue — investigate."
        ),
        "formula": "C = A + \\Phi",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_CORE,
    },

    "math_check_robustness": {
        "label": "R = -α·ln(α) Check",
        "tooltip": (
            "Verifies the robustness formula is correctly computed "
            "from relative ascendency."
        ),
        "definition": (
            "A consistency check ensuring that Robustness R equals "
            "-α·ln(α) within numerical precision."
        ),
        "interpret": (
            "**✅ Valid**: Robustness correctly derived from α. "
            "**⚠️ Error**: Formula implementation issue."
        ),
        "formula": "R = -\\alpha \\cdot \\ln(\\alpha)",
        "citation": REF_ULANOWICZ_2009,
        "doi": DOI_ULANOWICZ_2009,
        "category": CAT_CORE,
    },

    "math_check_tst": {
        "label": "TST Check",
        "tooltip": (
            "Verifies that Total System Throughput matches the sum "
            "of all flow matrix entries."
        ),
        "definition": (
            "A consistency check ensuring TST = Σ T_ij matches the "
            "reported value."
        ),
        "interpret": (
            "**✅ Valid**: TST correctly computed. "
            "**⚠️ Error**: Matrix summation issue."
        ),
        "category": CAT_CORE,
    },

    # =====================================================================
    # AUTOCATALYTIC INDEX
    # =====================================================================
    "autocatalytic_index": {
        "label": "Autocatalytic Index",
        "tooltip": (
            "Detects self-reinforcing feedback cycles — higher values "
            "mean stronger positive feedback loops."
        ),
        "definition": (
            "The Autocatalytic Index identifies and quantifies positive "
            "feedback cycles in the network. These are self-reinforcing "
            "loops where outputs feed back to strengthen inputs, driving "
            "growth and learning."
        ),
        "interpret": (
            "**High (>0.5)**: Strong self-reinforcing cycles — rapid "
            "learning and growth potential. "
            "**Low (<0.1)**: Few feedback loops — limited self-reinforcement."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Autonomous",
        "category": CAT_REGEN,
    },

    # =====================================================================
    # OASIS COMPONENT METRICS (sub-metrics shown in dimension details)
    # =====================================================================
    "oasis_reciprocity": {
        "label": "Reciprocity",
        "tooltip": (
            "Fraction of connections that are bidirectional — measures "
            "mutual exchange between nodes."
        ),
        "definition": (
            "The fraction of directed edges that have a reciprocal edge. "
            "Used as a component of the AUTONOMOUS dimension score."
        ),
        "interpret": (
            "**High**: Strong mutual exchange. "
            "**Low**: One-way flows dominate."
        ),
        "oasis_map": "Autonomous",
        "category": CAT_OASIS,
    },

    "oasis_mutualism": {
        "label": "Mutualism Index",
        "tooltip": (
            "Ratio of reciprocal to total connected pairs — measures "
            "cooperative vs. exploitative relationships."
        ),
        "definition": (
            "The mutualism ratio classifies pairwise relationships as "
            "mutualistic (bidirectional flow), exploitative (one-way), "
            "or neutral. Used in the SYMBIOTIC dimension."
        ),
        "interpret": (
            "**High (>0.5)**: Cooperative network — most relationships "
            "benefit both parties. "
            "**Low (<0.2)**: Extractive — mostly one-sided relationships."
        ),
        "citation": REF_FATH_2019,
        "doi": DOI_FATH_2019,
        "oasis_map": "Symbiotic",
        "category": CAT_OASIS,
    },

    "oasis_alpha_optimality": {
        "label": "Alpha Optimality",
        "tooltip": (
            "How close the efficiency ratio (α) is to the optimal "
            "0.37 — higher means closer to peak sustainability."
        ),
        "definition": (
            "Alpha Optimality = 1 - |α - 0.37|. Measures proximity "
            "to the theoretical optimum where robustness is maximized."
        ),
        "interpret": (
            "**Near 1.0**: At or very near the optimal balance. "
            "**Near 0**: Far from optimal — significant rebalancing needed."
        ),
        "oasis_map": "Sustainable",
        "category": CAT_OASIS,
    },

    "oasis_window_status": {
        "label": "Window Status",
        "tooltip": (
            "Whether the system is inside (✅) or outside (❌) the "
            "Window of Viability."
        ),
        "definition": (
            "Binary indicator: is 0.2 ≤ α ≤ 0.6? Used as a component "
            "of the SUSTAINABLE dimension score."
        ),
        "interpret": (
            "**✅ In Window**: System is viable. "
            "**❌ Outside**: Needs urgent rebalancing."
        ),
        "oasis_map": "Sustainable",
        "category": CAT_OASIS,
    },
}


# ---------------------------------------------------------------------------
# Helper: get all categories in display order
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [
    CAT_CORE,
    CAT_INFO,
    CAT_SUSTAIN,
    CAT_REGEN,
    CAT_OASIS,
    CAT_ROLES,
    CAT_NETWORK,
    CAT_FLOW,
    CAT_CENTRALITY,
    CAT_COMMUNITY,
    CAT_ROBUST,
    CAT_EXTENDED,
    CAT_BALANCE,
    CAT_HEALTH,
    CAT_VIZ,
    CAT_ECOSYSTEM,
]


def get_anchor(key: str) -> str:
    """Return the HTML anchor id for a registry key."""
    entry = DOCS.get(key, {})
    return entry.get("anchor", f"metric-{key.replace('_', '-')}")


def get_entries_by_category() -> Dict[str, list]:
    """Group all registry entries by category, in display order."""
    grouped: Dict[str, list] = {}
    for cat in CATEGORY_ORDER:
        grouped[cat] = []
    for key, entry in DOCS.items():
        cat = entry.get("category", "Other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append((key, entry))
    # Remove empty categories
    return {k: v for k, v in grouped.items() if v}
