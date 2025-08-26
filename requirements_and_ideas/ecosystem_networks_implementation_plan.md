# Ecosystem Networks Implementation Action Plan
## Adding New Sample Data and Validation Notebooks

*Created: January 2025*  
*Status: Ready for Implementation*  
*Priority: High*

---

## 📋 Executive Summary

This action plan outlines the systematic implementation of new ecosystem reference networks identified in our research, including data acquisition, processing, validation notebook creation, and integration into the Adaptive Organization Analysis System.

---

## 🎯 Implementation Phases Overview

```
Phase 1 (Week 1-2): High-Priority Networks
├── Allesina Food Web Networks
├── Sarafu Digital Currency Network
└── Basic Validation Notebooks

Phase 2 (Week 3-4): Extended Networks
├── Additional Ulanowicz Ecosystems
├── Brian Fath NEA.m Datasets
└── Comprehensive Validation Suite

Phase 3 (Week 5-6): Integration & Documentation
├── App Integration
├── Documentation Updates
└── Performance Testing
```

---

## 📊 Phase 1: High-Priority Networks (Weeks 1-2)

### **1.1 Allesina Food Web Networks**

#### Day 1-3: Data Acquisition & Processing

**Actions:**
```bash
# 1. Clone Allesina's repository
git clone https://github.com/StefanoAllesina/blockstructure.git temp/allesina_data

# 2. Review available networks
cd temp/allesina_data
ls -la data/
```

**Processing Steps:**
1. **Identify food web matrices** in the repository
2. **Extract flow matrices** from MATLAB/R/CSV formats
3. **Document network properties:**
   - Number of nodes (species)
   - Network type (marine, terrestrial, freshwater)
   - Original publication source
   - Published metrics (if available)

4. **Convert to JSON format:**
```python
# Script: convert_allesina_networks.py
import numpy as np
import json
import scipy.io as sio

def convert_foodweb_to_json(matlab_file, network_name, network_type):
    """Convert MATLAB food web to our JSON format."""
    
    # Load MATLAB file
    data = sio.loadmat(matlab_file)
    flow_matrix = data['flow_matrix']  # Adjust key name as needed
    species_names = data['species']  # Adjust key name as needed
    
    # Create JSON structure
    network_data = {
        "organization": f"{network_name} Food Web",
        "source": {
            "paper": "Allesina & Tang (2012) or specific paper",
            "authors": ["Stefano Allesina", "Co-authors"],
            "year": 2012,
            "doi": "10.xxxx/xxxxx",
            "github": "https://github.com/StefanoAllesina/blockstructure"
        },
        "nodes": species_names.tolist(),
        "flows": flow_matrix.tolist(),
        "metadata": {
            "units": "biomass flow or energy flow",
            "location": "Geographic location",
            "type": "ecosystem",
            "subtype": network_type,
            "collection_period": "YYYY-YYYY"
        },
        "validation": {
            "published_metrics": {
                "connectance": None,
                "modularity": None,
                "nestedness": None
            }
        }
    }
    
    # Save to file
    filename = f"allesina_{network_name.lower().replace(' ', '_')}.json"
    with open(f"data/ecosystem_samples/{filename}", 'w') as f:
        json.dump(network_data, f, indent=2)
    
    return filename
```

**Expected Outputs:**
- [ ] `allesina_marine_foodweb.json`
- [ ] `allesina_terrestrial_foodweb.json`
- [ ] `allesina_stream_foodweb.json`
- [ ] At least 3-5 food web networks

#### Day 4-5: Validation Notebook Creation

**Create:** `validation/allesina_foodwebs_validation.ipynb`

**Notebook Structure:**
```python
# Cell 1: Introduction
"""
# Allesina Food Web Networks Validation
## Testing Network Stability and Structure Metrics

This notebook validates our implementation against published food web 
metrics from Stefano Allesina's research on ecological network stability.
"""

# Cell 2: Load Networks
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')
from src.ulanowicz_calculator import UlanowiczCalculator

# Load all Allesina networks
networks = {}
data_path = Path('../data/ecosystem_samples')
for file in data_path.glob('allesina_*.json'):
    with open(file) as f:
        network_name = file.stem.replace('allesina_', '')
        networks[network_name] = json.load(f)

# Cell 3: Calculate Metrics for Each Network
results = {}
for name, network_data in networks.items():
    flows = np.array(network_data['flows'])
    nodes = network_data['nodes']
    
    calc = UlanowiczCalculator(flows, nodes)
    metrics = calc.get_extended_metrics()
    
    results[name] = {
        'tst': metrics['total_system_throughput'],
        'alpha': metrics['relative_ascendency'],
        'robustness': metrics['robustness'],
        'nodes': len(nodes),
        'connectance': np.count_nonzero(flows) / (len(nodes) ** 2)
    }

# Cell 4: Comparison with Published Metrics
# Compare calculated vs published metrics where available

# Cell 5: Stability Analysis
# Implement Allesina's stability criteria

# Cell 6: Visualization
# Create network visualizations and Window of Viability plots
```

---

### **1.2 Sarafu Digital Currency Network**

#### Day 6-8: Data Acquisition & Processing

**Actions:**
1. **Download the 2023 Scientific Reports paper**
2. **Access supplementary data** (transaction dataset)
3. **Process transaction data into flow matrix**

**Processing Script:**
```python
# Script: process_sarafu_network.py
import pandas as pd
import numpy as np
import json
from datetime import datetime

def process_sarafu_transactions(transaction_file):
    """Convert Sarafu transactions to flow network."""
    
    # Load transaction data
    df = pd.read_csv(transaction_file)
    
    # Aggregate by time period (e.g., monthly)
    df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
    
    # Create user categories or geographic regions as nodes
    # Option 1: Geographic regions
    regions = df['region'].unique()
    
    # Option 2: User types (merchants, consumers, etc.)
    user_types = df['user_type'].unique()
    
    # Build flow matrix
    flow_matrix = np.zeros((len(regions), len(regions)))
    
    for _, transaction in df.iterrows():
        from_idx = np.where(regions == transaction['from_region'])[0][0]
        to_idx = np.where(regions == transaction['to_region'])[0][0]
        flow_matrix[from_idx, to_idx] += transaction['amount']
    
    # Create JSON structure
    sarafu_network = {
        "organization": "Sarafu Community Currency Network",
        "source": {
            "paper": "Circulation of a digital community currency",
            "journal": "Scientific Reports",
            "authors": ["Author names from paper"],
            "year": 2023,
            "doi": "10.1038/xxxxx"
        },
        "nodes": regions.tolist(),
        "flows": flow_matrix.tolist(),
        "metadata": {
            "units": "Sarafu tokens",
            "location": "Kenya",
            "type": "economic",
            "subtype": "complementary_currency",
            "time_period": "2020-01 to 2021-06",
            "total_users": 40000,
            "total_volume": 293700000
        },
        "validation": {
            "network_properties": {
                "total_transactions": len(df),
                "active_users": df['user_id'].nunique(),
                "average_transaction": df['amount'].mean()
            }
        }
    }
    
    # Save variants (monthly snapshots)
    variants = ['2020_q1', '2020_q2', '2020_q3', '2020_q4', '2021_q1', '2021_q2']
    for variant in variants:
        # Filter and process each time period
        # Save each as separate network for temporal analysis
        pass
    
    return sarafu_network
```

**Expected Outputs:**
- [ ] `sarafu_full_network.json` (complete 2020-2021)
- [ ] `sarafu_2020_q1.json` through `sarafu_2021_q2.json` (quarterly snapshots)
- [ ] `sarafu_crisis_period.json` (COVID-19 period analysis)

#### Day 9-10: Validation Notebook Creation

**Create:** `validation/sarafu_currency_validation.ipynb`

**Notebook Structure:**
```python
# Cell 1: Introduction
"""
# Sarafu Digital Currency Network Validation
## Complementary Currency Flow Analysis (2020-2021)

Analyzing 293.7 million Sarafu transactions among 40,000 users
Based on Lietaer's monetary ecology framework
"""

# Cell 2: Load Network Data
# Load full network and quarterly snapshots

# Cell 3: Temporal Analysis
# Track how metrics change over time
# Compare pre-COVID, during-COVID, post-COVID periods

# Cell 4: Lietaer Framework Validation
# Calculate efficiency-resilience trade-off
# Compare with theoretical predictions

# Cell 5: Economic Insights
# Velocity of money calculations
# Network centrality analysis
# Community detection

# Cell 6: Visualization
# Sankey diagrams of money flow
# Temporal evolution plots
# Window of Viability trajectory
```

---

## 📊 Phase 2: Extended Networks (Weeks 3-4)

### **2.1 Additional Ulanowicz Ecosystems**

#### Day 11-13: Chesapeake Bay & Other Networks

**Search Sources:**
1. ResearchGate - Ulanowicz profile
2. Chesapeake Biological Laboratory data repository
3. Published papers with supplementary data

**Target Networks:**
- [ ] Chesapeake Bay mesohaline network
- [ ] St. Marks River ecosystem
- [ ] Crystal River Creek (Control)
- [ ] Florida Bay ecosystem

**Processing Template:**
```python
# Script: process_ulanowicz_legacy.py
def process_ulanowicz_network(source_file, network_name):
    """Process legacy Ulanowicz network data."""
    
    # Handle various formats (SCOR, EcoPath, custom)
    # Convert to standardized JSON
    pass
```

### **2.2 Brian Fath NEA.m Datasets**

#### Day 14-16: MATLAB Integration

**Actions:**
1. **Install NEA.m from MATLAB Central**
2. **Process existing networks through NEA.m**
3. **Extract additional metrics**

**MATLAB Script:**
```matlab
% process_with_neam.m
function results = process_with_neam(flow_matrix, node_names)
    % Run NEA.m analysis
    [stats, environ] = nea(flow_matrix);
    
    % Save results for Python import
    save('nea_results.mat', 'stats', 'environ');
    
    % Export to JSON-compatible format
    export_to_json(stats, 'nea_metrics.json');
end
```

### **2.3 Create Meta-Validation Notebook**

**Create:** `validation/ecosystem_networks_comparison.ipynb`

**Purpose:** Compare all ecosystem networks in one place

**Structure:**
```python
# Comprehensive comparison of all ecosystem networks
# - Prawns-Alligator (3 variants)
# - Cone Spring (2 variants)  
# - Allesina Food Webs (3-5 networks)
# - Sarafu Currency (6 time periods)
# - Ulanowicz Legacy (4+ networks)

# Generate master comparison table
# Create unified Window of Viability plot with all networks
# Statistical analysis of metric distributions
```

---

## 📊 Phase 3: Integration & Documentation (Weeks 5-6)

### **3.1 Update app.py**

#### Day 17-18: Streamlit Integration

**Modifications Required:**

1. **Update sample data section:**
```python
# In app.py, update the sample data interface

def load_sample_data():
    """Enhanced sample data loader with categories."""
    
    sample_categories = {
        "Classic Ecosystems": [
            "prawns_alligator_original",
            "prawns_alligator_adapted",
            "prawns_alligator_efficient",
            "cone_spring_original",
            "cone_spring_eutrophicated"
        ],
        "Food Web Networks": [
            "allesina_marine_foodweb",
            "allesina_terrestrial_foodweb",
            "allesina_stream_foodweb"
        ],
        "Economic Networks": [
            "sarafu_full_network",
            "sarafu_2020_q1",
            "sarafu_2021_q2"
        ],
        "Legacy Ulanowicz": [
            "chesapeake_bay",
            "st_marks_river",
            "crystal_river"
        ]
    }
    
    return sample_categories
```

2. **Add network info display:**
```python
def display_network_info(network_name):
    """Show source, validation status, and key metrics."""
    
    info = {
        'source_paper': network_data['source']['paper'],
        'authors': network_data['source']['authors'],
        'year': network_data['source']['year'],
        'nodes': len(network_data['nodes']),
        'validation_status': '✅ Validated' if validated else '⏳ Pending'
    }
    
    return info
```

### **3.2 Update Validation Notebooks Section**

#### Day 19-20: Extend Validation Interface

**Add to app.py:**
```python
# Update validation_notebooks_interface()

tabs = st.tabs([
    "🦐 Prawns-Alligator", 
    "🌿 Cone Spring",
    "🕸️ Food Webs",  # NEW
    "💰 Sarafu Currency",  # NEW
    "🌊 Chesapeake Bay",  # NEW
    "📊 Comparative Analysis"  # NEW
])
```

### **3.3 Documentation Updates**

#### Day 21-22: Update All Documentation

**Files to Update:**

1. **README.md**
```markdown
## 📚 Ecosystem Sample Networks

### Classic Networks (Validated)
- Prawns-Alligator (3 variants) - Ulanowicz et al. 2009
- Cone Spring (2 variants) - Eutrophication study

### Food Web Networks (NEW)
- Marine Food Web - Allesina & Tang 2012
- Terrestrial Food Web - [Source]
- Stream Food Web - [Source]

### Economic Networks (NEW)  
- Sarafu Digital Currency - 40,000 users, Kenya 2020-2021
- Temporal snapshots for crisis analysis

### Legacy Ecosystems
- Chesapeake Bay - Ulanowicz classic study
- Florida Bay - Ecosystem analysis
```

2. **Create:** `docs/ecosystem_networks_catalog.md`
```markdown
# Complete Catalog of Ecosystem Networks

## Network Properties Summary Table
| Network | Nodes | Type | TST | Alpha | Status | Source |
|---------|-------|------|-----|-------|--------|--------|
| ... comprehensive table ... |
```

3. **Create:** `validation/README.md`
```markdown
# Validation Notebooks Guide

## Available Notebooks
1. prawns_alligator_validation.ipynb - Classic 3-network comparison
2. cone_spring_validation.ipynb - Eutrophication effects
3. allesina_foodwebs_validation.ipynb - Food web stability
4. sarafu_currency_validation.ipynb - Economic network analysis
5. ecosystem_networks_comparison.ipynb - Meta-analysis
```

---

## 📋 Task Checklist

### Week 1-2: Priority Implementation
- [ ] Clone and process Allesina GitHub repository
- [ ] Create 3-5 food web JSON files
- [ ] Build Allesina validation notebook
- [ ] Download Sarafu paper and data
- [ ] Process Sarafu transaction network
- [ ] Create Sarafu validation notebook
- [ ] Test both notebooks thoroughly

### Week 3-4: Extended Networks
- [ ] Contact researchers for additional datasets
- [ ] Process Chesapeake Bay network
- [ ] Add Florida Bay ecosystem
- [ ] Install and test NEA.m toolkit
- [ ] Create comparative analysis notebook
- [ ] Validate all calculations

### Week 5-6: Integration
- [ ] Update app.py with new categories
- [ ] Enhance validation notebooks interface
- [ ] Update README.md
- [ ] Create ecosystem catalog document
- [ ] Write validation guide
- [ ] Performance testing
- [ ] Final review and testing

---

## 🎯 Success Metrics

### Quantitative Goals:
- **15+ new ecosystem networks** implemented
- **5+ validation notebooks** created
- **100% validation** against published metrics where available
- **<5% error rate** on known metrics

### Quality Standards:
- All networks properly documented with sources
- JSON format consistent across all networks
- Validation notebooks follow standard template
- Clear visualization for each network
- Performance maintained with larger dataset

---

## 🚀 Quick Start Commands

```bash
# Set up workspace
mkdir -p temp/network_sources
cd temp/network_sources

# Clone Allesina repository
git clone https://github.com/StefanoAllesina/blockstructure.git

# Create processing scripts directory
mkdir -p ../../scripts/network_processing
cd ../../scripts/network_processing

# Run processing pipeline
python process_allesina_networks.py
python process_sarafu_network.py
python process_ulanowicz_legacy.py

# Test validation notebooks
jupyter notebook validation/

# Run integration tests
python tests/test_new_networks.py
```

---

## 📝 Notes & Considerations

### Data Quality:
- Verify units consistency (energy vs biomass vs currency)
- Document any assumptions in conversions
- Note temporal aspects (snapshot vs aggregated)
- Handle missing data appropriately

### Validation Priorities:
1. Networks with published metrics (highest priority)
2. Networks with partial metrics
3. Networks for demonstration only

### Performance Considerations:
- Large networks (>100 nodes) may need optimization
- Consider lazy loading for app interface
- Cache calculated metrics for quick access

### Future Extensions:
- Temporal network evolution animations
- Interactive network editor
- Batch comparison tools
- API for programmatic access

---

## 📞 Support & Resources

### Key Contacts:
- Stefano Allesina: [University email]
- Brian Fath: [Contact via ResearchGate]
- Research Alliance for Regenerative Economics

### Technical Resources:
- Ulanowicz Calculator Documentation
- Network Analysis Best Practices
- JSON Schema Validation

### Community:
- GitHub Issues for bug reports
- Discussion forum for validation questions
- Slack channel for real-time support

---

*This action plan provides a systematic approach to expanding the Adaptive Organization Analysis System with scientifically validated ecosystem networks.*

*Version: 1.0 | Created: January 2025*