# OASIS - Organizational Adaptive Sustainability Intelligence System

**Analyze organizational sustainability through the lens of ecosystem theory, using Ulanowicz's information-theoretic framework and the OASIS health assessment model.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.45+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

OASIS treats organizations as living ecosystems — networks of information, resource, and communication flows between departments, teams, or entities. By applying Robert Ulanowicz's information-theoretic measures from network ecology, it quantifies organizational health across five dimensions: **Open, Autonomous, Symbiotic, Intelligent, and Sustainable**.

The system provides:

- **Network flow analysis** using Ulanowicz metrics (TST, AMI, Ascendency, Robustness)
- **OASIS health assessment** with five-dimension scoring and actionable recommendations
- **Professional PDF reports** structured to Big Four audit-firm standards
- **Interactive dark-mode dashboard** with nature/ecosystem visual theme
- **Real-world dataset library** (airports, supply chains, energy grids, trade networks)
- **Synthetic data generation** for testing and research

## What's New in v2.0.0

- **Dark mode UI** — Full nature/ecosystem color palette (forest green, gold, teal) across all components
- **Professional PDF reports** — Cover page, table of contents, embedded charts, headers/footers via reportlab
- **Publication-quality report narratives** — Restructured to Big Four audit-firm standards with narrative findings
- **Interactive documentation system** — In-app tooltips and reference documentation for all metrics
- **OASIS health model** — Five-dimension organizational health scoring with traffic-light indicators
- **SQLite metrics database** — Precomputed metrics for faster analysis of saved networks
- **HuggingFace dataset discovery** — AI agent for finding relevant network datasets
- **Ecosystem sample datasets** — US airports, Bitcoin transactions, pharma, manufacturing networks

## Quick Start

```bash
# Clone the repository
git clone https://github.com/maxdolphin/OASIS.git
cd OASIS

# Install dependencies
pip install -r docs/requirements.txt

# Launch the app
streamlit run app.py
```

Open your browser to **http://localhost:8501**

### Python API

```python
import numpy as np
from src.ulanowicz_calculator import UlanowiczCalculator

flow_matrix = np.array([
    [0, 8, 6, 4],
    [5, 0, 7, 3],
    [4, 6, 0, 5],
    [3, 4, 5, 0]
])
departments = ['Sales', 'Marketing', 'Operations', 'Support']

calc = UlanowiczCalculator(flow_matrix, departments)
metrics = calc.get_extended_metrics()

print(f"Robustness: {metrics['robustness']:.3f}")
print(f"Viable: {'YES' if metrics['is_viable'] else 'NO'}")
```

## Key Features

### Ulanowicz Metrics
- Total System Throughput (TST), Average Mutual Information (AMI)
- Ascendency (A), Development Capacity (C), Overhead (Phi)
- Robustness, Window of Viability analysis
- Flow Diversity, Redundancy, Regenerative Capacity
- Trophic Depth, Network Efficiency, Structural Information

### OASIS Health Assessment
Each organization is scored across five ecosystem-inspired dimensions:
- **Open** — Connectivity and exchange capacity
- **Autonomous** — Self-governance and adaptive capacity
- **Symbiotic** — Mutualistic relationships and cooperation
- **Intelligent** — Information processing and flow efficiency
- **Sustainable** — Long-term viability and resilience

### Visualizations
- Robustness curve with Window of Viability
- Sankey flow diagrams and network heatmaps
- OASIS radar chart and dimension gauges
- Network topology graphs
- All charts interactive (Plotly) with dark theme

### Professional Reporting
- PDF export with cover page, ToC, and embedded charts
- Narrative findings structured per Big Four audit conventions
- OASIS dimension assessments with framework alignment (ESRS, GRI, TCFD)
- Prioritized recommendations (Immediate / Short-term / Medium-term)

### Data Sources
- Upload your own data (JSON or CSV flow matrices)
- Generate synthetic organizations with configurable parameters
- Built-in ecosystem samples: US airports, Bitcoin, pharma, manufacturing
- Real-world datasets: energy grids, supply chains, financial networks, trade flows
- HuggingFace dataset discovery agent

## Project Structure

```
OASIS/
├── app.py                          # Streamlit web application
├── .streamlit/config.toml          # Dark theme configuration
├── src/
│   ├── ulanowicz_calculator.py     # Core Ulanowicz metrics
│   ├── oasis_calculator.py         # OASIS health assessment
│   ├── oasis_visualizer.py         # OASIS charts and gauges
│   ├── oasis_report.py             # OASIS narrative report
│   ├── visualizer.py               # Network visualizations
│   ├── network_generator.py        # Synthetic data generation
│   ├── pdf_generator.py            # Professional PDF export
│   ├── publication_report.py       # Publication-quality narratives
│   ├── docs_registry.py            # Documentation content registry
│   ├── docs_ui.py                  # In-app documentation UI
│   ├── database/                   # SQLite metrics storage
│   └── services/                   # Validation and metrics services
├── data/
│   ├── ecosystem_samples/          # Real-world network datasets
│   ├── user_saved_networks/        # Saved analysis networks
│   └── synthetic_organizations/    # Generated test data
├── papers/                         # Scientific references
├── docs/                           # Documentation and requirements
└── tests/                          # Test suite
```

## Theoretical Foundation

Based on the work of:

- **Robert E. Ulanowicz** — Ecosystem sustainability theory, Ascendency concept, Window of Viability
- **Brian D. Fath & Robert E. Ulanowicz** — "Measuring Regenerative Economics: 10 principles and measures undergirding systemic economic health" (2019)
- **Bernard C. Patten** — Network environ analysis, indirect effects in ecosystems
- **Stuart Kauffman** — Self-organization, edge of chaos in complex systems

### Key Concepts
- **Window of Viability**: Sustainable systems operate between 20-60% efficiency ratio (A/C)
- **Robustness**: Optimal balance at approximately 37% efficiency, following R = -a * log(a)
- **Ascendency vs Overhead**: The tension between organized efficiency and adaptive reserve capacity

## Data Formats

### JSON
```json
{
  "organization": "My Company",
  "nodes": ["Sales", "Marketing", "Operations", "Support"],
  "flows": [
    [0.0, 8.0, 6.0, 4.0],
    [5.0, 0.0, 7.0, 3.0],
    [4.0, 6.0, 0.0, 5.0],
    [3.0, 4.0, 5.0, 0.0]
  ]
}
```

### CSV
```csv
,Sales,Marketing,Operations,Support
Sales,0.0,8.0,6.0,4.0
Marketing,5.0,0.0,7.0,3.0
Operations,4.0,6.0,0.0,5.0
Support,3.0,4.0,5.0,0.0
```

## Requirements

- Python 3.8+
- 8GB RAM recommended for large networks

Install all dependencies:
```bash
pip install -r docs/requirements.txt
```

## Contributing

Contributions welcome — report bugs, suggest features, improve documentation, or add analysis methods via GitHub issues and pull requests.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Robert E. Ulanowicz** for the foundational ecosystem theory
- **Brian D. Fath** for extending the framework to regenerative economics
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations

---

```bash
git clone https://github.com/maxdolphin/OASIS.git
cd OASIS && streamlit run app.py
```
