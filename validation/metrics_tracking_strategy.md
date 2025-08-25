# Metrics Tracking Strategy - Implementation Plan

## Overview
Create a persistent, structured system to track published metrics from papers alongside our calculated values for comprehensive validation and comparison.

## Data Structure Design

### 1. Core Metrics to Track

#### Published Metrics (from papers):
- **TST** (Total System Throughput)
- **C** (Development Capacity) 
- **A** (Ascendency)
- **Φ** (Reserve/Overhead)
- **α** (Relative Ascendency)
- **R** (Robustness)
- **AMI** (Average Mutual Information)
- **H** (Flow Diversity)
- **ξ** (Effective Connectivity)
- **η** (Effective Trophic Levels)

#### Calculated Metrics (our implementation):
- All of the above, calculated using our formulas
- Additional derived metrics
- Error percentages
- Validation status

### 2. Data Structure Schema

```python
{
    "network_id": "prawns_alligator_adapted",
    "network_name": "Prawns-Alligator Ecosystem (After Fish Loss)",
    "source": {
        "paper": "Ulanowicz et al. 2009",
        "doi": "10.1016/j.ecocom.2008.10.005",
        "figure": "Figure 3",
        "page": 31
    },
    "network_data": {
        "nodes": ["Prawns", "Turtles", "Snakes", "Alligators"],
        "flows": [[0, 74.3, 16.1, 0], ...],
        "units": "mg C m⁻² y⁻¹"
    },
    "published_metrics": {
        "tst": {"value": 99.7, "unit": "mg C m⁻² y⁻¹", "reported": true},
        "development_capacity": {"value": null, "unit": "mg C-bits m⁻² y⁻¹", "reported": false},
        "ascendency": {"value": 44.5, "unit": "mg C-bits m⁻² y⁻¹", "reported": true},
        "reserve": {"value": 68.2, "unit": "mg C-bits m⁻² y⁻¹", "reported": true},
        "relative_ascendency": {"value": null, "unit": "dimensionless", "reported": false},
        "robustness": {"value": null, "unit": "dimensionless", "reported": false},
        "ami": {"value": null, "unit": "bits", "reported": false},
        "flow_diversity": {"value": null, "unit": "bits", "reported": false}
    },
    "calculated_metrics": {
        "tst": {"value": null, "unit": "mg C m⁻² y⁻¹"},
        "development_capacity": {"value": null, "unit": "mg C-bits m⁻² y⁻¹"},
        "ascendency": {"value": null, "unit": "mg C-bits m⁻² y⁻¹"},
        "reserve": {"value": null, "unit": "mg C-bits m⁻² y⁻¹"},
        "relative_ascendency": {"value": null, "unit": "dimensionless"},
        "robustness": {"value": null, "unit": "dimensionless"},
        "ami": {"value": null, "unit": "bits"},
        "flow_diversity": {"value": null, "unit": "bits"}
    },
    "comparison": {
        "tst_error": null,
        "ascendency_error": null,
        "reserve_error": null,
        "alpha_error": null,
        "fundamental_check": null,
        "validation_status": "pending"
    },
    "metadata": {
        "date_extracted": "2025-08-25",
        "extracted_by": "validation_v2.1.4",
        "notes": "System after fish loss, demonstrates resilience"
    }
}
```

### 3. Storage Options

#### Option A: JSON Database (Recommended for v2.1.4)
```
validation/
  metrics_database/
    master_registry.json      # List of all networks
    networks/
      cone_spring_original.json
      cone_spring_eutrophicated.json
      prawns_alligator_original.json
      prawns_alligator_efficient.json
      prawns_alligator_adapted.json
    reports/
      comparison_report.json
      validation_summary.json
```

#### Option B: SQLite Database (Future v2.2.0)
```sql
CREATE TABLE networks (
    id TEXT PRIMARY KEY,
    name TEXT,
    source_paper TEXT,
    doi TEXT,
    figure TEXT,
    page INTEGER,
    nodes JSON,
    flows JSON,
    units TEXT
);

CREATE TABLE published_metrics (
    network_id TEXT,
    metric_name TEXT,
    value REAL,
    unit TEXT,
    reported BOOLEAN,
    FOREIGN KEY (network_id) REFERENCES networks(id)
);

CREATE TABLE calculated_metrics (
    network_id TEXT,
    metric_name TEXT,
    value REAL,
    unit TEXT,
    calculation_date TIMESTAMP,
    FOREIGN KEY (network_id) REFERENCES networks(id)
);
```

### 4. Implementation Strategy

#### Phase 1: Data Structure Setup (Current)
1. Create metrics database directory structure
2. Define standardized metric names and units
3. Create template for network metrics
4. Build registry of all networks

#### Phase 2: Data Population
1. Extract all published metrics from papers
2. Tag unreported metrics as N/A
3. Create structured JSON files for each network
4. Build master registry

#### Phase 3: Calculation Framework
1. Create batch calculator for all networks
2. Populate calculated_metrics fields
3. Compute error percentages
4. Generate validation status

#### Phase 4: Comparison & Reporting
1. Build comparison dashboard
2. Create validation reports
3. Generate discrepancy analysis
4. Export to various formats

### 5. Key Functions to Implement

```python
class MetricsTracker:
    def __init__(self):
        self.database_path = "validation/metrics_database"
        self.networks = {}
    
    def load_network(self, network_id):
        """Load network metrics from database"""
        
    def add_published_metric(self, network_id, metric_name, value, unit):
        """Add a published metric from paper"""
        
    def calculate_metrics(self, network_id):
        """Calculate all metrics using our formulas"""
        
    def compare_metrics(self, network_id):
        """Compare published vs calculated"""
        
    def generate_report(self):
        """Generate comprehensive validation report"""
        
    def export_to_csv(self):
        """Export comparison table to CSV"""
```

### 6. Standardized Metric Names

To ensure consistency across all networks:

```python
STANDARD_METRICS = {
    'tst': 'Total System Throughput',
    'development_capacity': 'Development Capacity (C)',
    'ascendency': 'Ascendency (A)',
    'reserve': 'Reserve/Overhead (Φ)',
    'relative_ascendency': 'Relative Ascendency (α)',
    'robustness': 'Robustness (R)',
    'ami': 'Average Mutual Information',
    'flow_diversity': 'Flow Diversity (H)',
    'effective_connectivity': 'Effective Connectivity (ξ)',
    'trophic_levels': 'Effective Trophic Levels (η)'
}
```

### 7. Benefits of This Approach

1. **Systematic Validation**: Track what's reported vs calculated
2. **Gap Analysis**: Identify which metrics papers don't report
3. **Error Tracking**: Quantify discrepancies systematically
4. **Reproducibility**: Document all sources and calculations
5. **Extensibility**: Easy to add new networks and metrics
6. **Reporting**: Generate comprehensive validation reports

### 8. Next Steps

1. **Immediate** (v2.1.4):
   - Create directory structure
   - Build JSON templates
   - Populate existing networks (Cone Spring, Prawns-Alligator)
   - Create basic comparison script

2. **Short-term** (v2.1.5):
   - Add more networks from papers
   - Build automated calculator
   - Generate first validation report

3. **Long-term** (v2.2.0):
   - Migrate to SQLite
   - Add web interface
   - Create API endpoints
   - Build interactive dashboard

### 9. Example Usage

```python
# Initialize tracker
tracker = MetricsTracker()

# Load network
tracker.load_network('prawns_alligator_adapted')

# Add published metrics
tracker.add_published_metric('prawns_alligator_adapted', 'tst', 99.7, 'mg C m⁻² y⁻¹')
tracker.add_published_metric('prawns_alligator_adapted', 'ascendency', 44.5, 'mg C-bits m⁻² y⁻¹')

# Calculate our metrics
tracker.calculate_metrics('prawns_alligator_adapted')

# Compare and report
comparison = tracker.compare_metrics('prawns_alligator_adapted')
print(comparison.summary())
```

---
*Strategy Document for Adaptive Organization Analysis System v2.1.4*
*Validation Enhancement - Metrics Tracking*