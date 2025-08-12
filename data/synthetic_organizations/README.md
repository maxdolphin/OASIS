# Synthetic Organizational Communication Data

This directory contains synthetic datasets representing realistic organizational communication patterns for testing and demonstrating the Adaptive Organization Analysis system.

## Directory Structure

```
synthetic_organizations/
├── organizational_structures/    # Organizational hierarchy and role definitions
├── email_flows/                 # Email exchange flow matrices
├── document_flows/              # Document sharing flow matrices  
├── combined_flows/              # Weighted combination of email + document flows
├── generate_synthetic_data.py   # Data generation script
└── README.md                    # This file
```

## Data Types

### 1. Organizational Structures (`organizational_structures/`)

Contains JSON files defining:
- Department hierarchies and roles
- Employee lists and external collaborators
- Communication frequency patterns
- Collaboration intensity mappings

**Example**: `ideal_tech_company.json`

### 2. Email Flow Matrices (`email_flows/`)

Monthly email exchange volumes between departments:
- **Units**: Emails per month
- **Pattern**: Higher frequency, lower strategic weight
- **Characteristics**: Daily operational communication

**Example**: `tech_company_email_matrix.json`

### 3. Document Flow Matrices (`document_flows/`)

Monthly document sharing patterns:
- **Units**: Documents shared per month  
- **Pattern**: Lower frequency, higher strategic weight
- **Characteristics**: Formal information transfer

**Example**: `tech_company_document_matrix.json`

### 4. Combined Flow Matrices (`combined_flows/`)

Weighted combination of email and document flows:
- **Formula**: `Combined = (Email × 0.6) + (Document × 1.4)`
- **Rationale**: Documents carry more strategic information weight
- **Use Case**: Primary input for regenerative economics analysis

**Example**: `tech_company_combined_matrix.json`

## Sample Organization: TechFlow Innovations

A 45-employee technology company with 10 departments:

| Department | Employees | Primary Role |
|------------|-----------|--------------|
| Executive | 3 | Strategic leadership and decision making |
| Product | 4 | Product strategy and design |
| Engineering | 8 | Software development and implementation |
| Data_Science | 3 | Data analysis and machine learning |
| Sales | 4 | Revenue generation and client relationships |
| Marketing | 3 | Brand awareness and lead generation |
| Customer_Success | 3 | Customer support and retention |
| Operations | 3 | Business operations and infrastructure |
| HR | 2 | Human resources and talent management |
| Finance | 2 | Financial planning and analysis |

### Communication Patterns

**Highest Flow Connections**:
- Sales ↔ Customer_Success (customer handoffs)
- Product ↔ Engineering (development collaboration)
- Marketing ↔ Sales (lead management)

**Key Hubs**:
- **Product**: Central coordination hub (514.0 total outflow)
- **Executive**: Strategic communication hub (348.2 total outflow)
- **Sales**: Customer-facing hub (388.0 total outflow)

## Using the Data

### 1. Direct Analysis

```python
# Load combined flow data
with open('combined_flows/tech_company_combined_matrix.json', 'r') as f:
    data = json.load(f)

flow_matrix = np.array(data['flows'])
departments = data['nodes']

# Analyze with Ulanowicz calculator
from src.ulanowicz_calculator import UlanowiczCalculator
calc = UlanowiczCalculator(flow_matrix, departments)
metrics = calc.get_extended_metrics()
```

### 2. Main Application

```bash
# Analyze the synthetic organization
python src/main.py --input data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json --detailed

# Generate visualizations
python src/main.py --input data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json --output analysis.html
```

### 3. Generate New Data

```bash
# Generate new organization with default settings
python data/synthetic_organizations/generate_synthetic_data.py --org-name "MyCompany"

# Generate with specific parameters
python data/synthetic_organizations/generate_synthetic_data.py \
    --org-name "HighTech Corp" \
    --intensity high \
    --formality high \
    --seed 123
```

## Data Generation Parameters

### Communication Intensity Levels

- **Low**: Base 15 emails/month, variance ±10
- **Medium**: Base 30 emails/month, variance ±20  
- **High**: Base 50 emails/month, variance ±30

### Document Formality Levels

- **Low**: Base 8 docs/month, admin multiplier 1.5x
- **Medium**: Base 15 docs/month, admin multiplier 2.0x
- **High**: Base 25 docs/month, admin multiplier 3.0x

## Validation and Realism

The synthetic data incorporates realistic organizational patterns:

1. **Hierarchical Communication**: Executive departments broadcast more
2. **Functional Clustering**: Related departments (Product-Engineering, Sales-Marketing) have higher flows
3. **Administrative Patterns**: HR, Finance, Operations generate procedural documents
4. **Bidirectional Asymmetries**: Realistic imbalances in communication flows
5. **External Collaborations**: Each department has appropriate external partners

## Regenerative Economics Analysis

The combined flow matrices are designed to demonstrate key regenerative economics principles:

- **Window of Viability**: TechFlow operates at ~0.39 efficiency ratio (within 0.2-0.6 range)
- **Robustness**: ~0.18 robustness score (moderate, room for improvement)
- **Flow Diversity**: Balanced distribution across communication channels
- **Network Efficiency**: Optimal balance between organization and flexibility

## Use Cases

1. **Algorithm Testing**: Validate Ulanowicz calculations with known patterns
2. **Benchmarking**: Compare real organizations against synthetic ideals
3. **Scenario Analysis**: Modify flows to test different organizational structures
4. **Training**: Demonstrate regenerative economics concepts with realistic data
5. **Research**: Study organizational communication network properties

## Future Extensions

Planned additions:
- Seasonal variation patterns
- Crisis/growth scenario datasets  
- Industry-specific organizational templates
- Multi-location/remote work patterns
- Project-based temporary flow increases