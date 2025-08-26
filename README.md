# 🌱 Adaptive Organization Analysis

**A comprehensive system for analyzing organizational sustainability using Ulanowicz's ecosystem theory and regenerative economics principles.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.48+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements the **Ulanowicz-Fath regenerative economics framework** to analyze organizational communication networks and assess sustainability. It provides tools to:

- 📊 **Analyze organizational flow networks** (email, documents, communications)
- 🌱 **Calculate regenerative economics indicators** (robustness, efficiency, resilience)  
- 🎯 **Visualize position in the window of viability** with interactive sustainability curves
- 📈 **Generate comprehensive reports** with actionable recommendations
- ⚡ **Create synthetic organizational data** for testing and research

## ✨ Key Features

### 🔬 **Complete Ulanowicz Implementation**
- Total System Throughput (TST)
- Average Mutual Information (AMI)
- Ascendency (A), Development Capacity (C), Overhead (Φ)
- Window of Viability analysis

### 🌱 **Extended Regenerative Economics**
- Flow Diversity & Structural Information
- Robustness & Network Efficiency
- Regenerative Capacity & Resilience measures
- Trophic Depth & Redundancy analysis

### 🌐 **Interactive Web Application**
- **Streamlit-based dashboard** with professional UI
- **File upload support** (JSON, CSV)
- **Synthetic data generator** with configurable parameters
- **Real-time analysis** and visualization
- **Export capabilities** (reports, data, visualizations)

### 📊 **Advanced Visualizations**
- **🎯 Sustainability Curve**: Shows organization position on theoretical curve with window of viability
- **💪 Robustness Analysis**: Efficiency vs. robustness relationship with optimal points
- **🔥 Network Heatmaps**: Department-to-department flow visualization
- All charts interactive with Plotly (zoom, pan, hover tooltips)

## 🚀 Quick Start

### **Method 1: Web Application (Recommended)**
```bash
# Clone the repository
git clone https://github.com/yourusername/Adaptive_Organization.git
cd Adaptive_Organization

# Install dependencies
pip install -r docs/requirements.txt

# Launch web app
./run_webapp.sh
# OR
streamlit run app.py
```

**Open your browser to http://localhost:8501**

### **Method 2: Command Line Interface**
```bash
# Analyze sample organization
python3 src/main.py --example --detailed

# Analyze your data
python3 src/main.py --input your_data.json --output analysis.html --detailed

# Generate synthetic data
python3 data/synthetic_organizations/generate_synthetic_data.py --org-name "MyCompany"
```

### **Method 3: Python API**
```python
import numpy as np
from src.ulanowicz_calculator import UlanowiczCalculator
from src.visualizer import SustainabilityVisualizer

# Your organizational flow matrix
flow_matrix = np.array([
    [0, 8, 6, 4],
    [5, 0, 7, 3], 
    [4, 6, 0, 5],
    [3, 4, 5, 0]
])
departments = ['Sales', 'Marketing', 'Operations', 'Support']

# Calculate sustainability metrics
calc = UlanowiczCalculator(flow_matrix, departments)
metrics = calc.get_extended_metrics()

print(f"Robustness: {metrics['robustness']:.3f}")
print(f"Viable: {'YES' if metrics['is_viable'] else 'NO'}")

# Generate visualizations
viz = SustainabilityVisualizer(calc)
viz.save_visualization('analysis.html', 'html')
```

## 📊 Example Results

### **TechFlow Innovations (Sample Organization)**
- **Robustness**: 0.587 (HIGH - Strong robustness)
- **Network Efficiency**: 0.066 (Low - Needs streamlining)  
- **Regenerative Capacity**: 0.409 (HIGH - Strong potential)
- **Status**: "Too chaotic" - Needs more organizational structure
- **Recommendation**: Increase coordination, reduce redundant pathways

## 📁 Project Structure

```
Adaptive_Organization/
├── 🌐 app.py                          # Streamlit web application
├── 🚀 run_webapp.sh                   # Quick launcher script
├── 📊 src/                            # Core implementation
│   ├── ulanowicz_calculator.py        # All sustainability calculations
│   ├── visualizer.py                  # Visualization system
│   └── main.py                        # CLI application
├── 📈 data/synthetic_organizations/   # Synthetic datasets
│   ├── email_flows/                   # Email communication matrices
│   ├── document_flows/                # Document sharing patterns
│   ├── combined_flows/                # Weighted combined flows
│   ├── organizational_structures/     # Company hierarchies
│   └── generate_synthetic_data.py     # Data generation script
├── 📚 docs/                           # Documentation & examples
│   ├── example_usage.py               # Basic usage examples
│   ├── extended_example.py            # Comprehensive demo
│   └── requirements.txt               # Dependencies
└── 📋 Documentation files             # Guides and summaries
```

## 🔧 Installation & Requirements

### **System Requirements**
- Python 3.8 or higher
- 8GB RAM recommended for large organizations  
- Modern web browser for Streamlit interface

### **Dependencies**
```bash
pip install streamlit numpy pandas plotly matplotlib networkx scipy
```

All dependencies are listed in `docs/requirements.txt` and will be installed automatically when using the setup scripts.

## 📊 Data Formats

### **JSON Format**
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

### **CSV Format**
```csv
,Sales,Marketing,Operations,Support
Sales,0.0,8.0,6.0,4.0
Marketing,5.0,0.0,7.0,3.0
Operations,4.0,6.0,0.0,5.0
Support,3.0,4.0,5.0,0.0
```

## 🎓 Theoretical Foundation

This implementation is based on:

- **Robert E. Ulanowicz** - Ecosystem Sustainability Theory, Ascendency concept
- **Brian D. Fath & Robert E. Ulanowicz** - "Measuring Regenerative Economics: 10 principles and measures undergirding systemic economic health" (2019)
- **Network Ecology** - Information theory applications to complex systems
- **Systems Theory** - Complex adaptive systems, resilience, and robustness

### **Key Concepts**
- **Window of Viability**: Sustainable systems operate between 20-60% efficiency ratio
- **Robustness**: Optimal balance occurs around 37% efficiency (A/C ratio)
- **Regenerative Capacity**: System's ability to self-renew and adapt to change
- **Flow Networks**: Organizations as networks of information and resource flows

## 🎨 Visualization Gallery

### **🎯 Sustainability Curve**
- Shows your organization's position on the theoretical sustainability curve
- Window of viability boundaries clearly marked
- Reference organizations for benchmarking
- Interactive exploration with detailed tooltips

### **💪 Robustness Analysis**  
- Efficiency vs. robustness relationship
- Optimal point identification
- Window of viability highlighting

### **🔥 Network Flow Analysis**
- Department-to-department communication heatmaps
- Flow intensity visualization
- Network structure analysis

## 💼 Use Cases

### **Business Applications**
- **Organizational Design**: Optimize department structure and communication flows
- **Change Management**: Assess impact of restructuring on system sustainability  
- **Performance Analysis**: Identify bottlenecks and inefficiencies
- **Merger & Acquisition**: Evaluate organizational compatibility and integration challenges

### **Research Applications**
- **Systems Thinking**: Demonstrate network effects in organizations
- **Sustainability Science**: Apply ecological principles to human systems
- **Management Science**: Quantify organizational health and resilience
- **Complexity Studies**: Analyze emergent properties in social networks

### **Educational Applications**
- **MBA Programs**: Teach systems thinking and organizational theory
- **Executive Education**: Demonstrate regenerative business principles
- **Consulting Training**: Provide tools for organizational analysis
- **Research Methods**: Quantitative approaches to organizational study

## 📈 Sample Organizations Included

- **TechFlow Innovations**: 45-employee technology company (10 departments)
- **Balanced Test Organization**: Small balanced system for learning
- **Email vs. Document Flows**: Separate analysis of communication types
- **Synthetic Data Generator**: Create custom organizations with realistic patterns

## 📚 Documentation

### Core Documentation
- **[Flow Datasets Documentation](docs/flow_datasets_documentation.md)** - Comprehensive guide to large-scale real-world datasets
- **[Flow Datasets Requirements](requirements_and_ideas/flow_datasets_requirements.md)** - Technical requirements and specifications
- **[Webapp Guide](WEBAPP_GUIDE.md)** - Complete application usage guide
- **[Project Summary](PROJECT_SUMMARY.md)** - High-level system overview

### 🚀 Real-World Datasets - Ready for Immediate Analysis
The system includes **9 fully processed, ready-to-use** flow datasets from multiple domains:

#### ⚡ Energy Flow Networks (2 datasets)
- **✅ European Power Grid Network** - 10 nodes, continental energy distribution (9,832 MW total flow)
- **✅ Smart Grid Real-Time Monitoring** - Smart grid monitoring system

#### 🏭 Supply Chain Networks (2 datasets)  
- **✅ DataCo Smart Supply Chain** - 12 nodes, multi-tier supply network (suppliers→retail)
- **✅ Logistics and Supply Chain Network** - Modern distribution system

#### 💰 Financial Flow Networks (2 datasets)
- **✅ PaySim Mobile Money Network** - 16 nodes, banking ecosystem with payment flows
- **✅ Banking Transaction Network** - Financial transaction system

#### 🌍 Trade & Material Networks (3 datasets)
- **✅ OECD Input-Output Network** - 15 economic sectors, international flows
- **✅ EU Material Flow Network** - 20 EU countries, material resource flows  
- **✅ WTO Global Trade Network** - 20 countries, international trade flows

**No conversion needed** - All datasets processed and ready for immediate Ulanowicz analysis!

### Validation & Research
- **[Validation Notebooks](validation/)** - Jupyter notebooks validating calculations
- **[Prawns-Alligator Analysis](validation/prawns_alligator_validation.ipynb)** - Ecosystem validation
- **[Research Papers](papers/)** - Scientific literature and references

### Technical Resources
- **[API Documentation](src/)** - Core calculation modules
- **[Requirements](docs/requirements.txt)** - System dependencies
- **[Contributing Guidelines](CONTRIBUTING.md)** - Development participation

Access these datasets through the **"🌍 Real Life Data"** section in the application's sample data interface.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📚 Improve documentation  
- 🔬 Add new analysis methods
- 🎨 Enhance visualizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Robert E. Ulanowicz** for the foundational ecosystem theory
- **Brian D. Fath** for extending the framework to economics
- **Streamlit team** for the excellent web framework
- **Plotly** for interactive visualizations
- **Scientific Python community** for the computational tools

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

## 🎯 Get Started Now!

```bash
git clone https://github.com/yourusername/Adaptive_Organization.git
cd Adaptive_Organization
./run_webapp.sh
```

**🌐 Open http://localhost:8501 and start analyzing organizational sustainability!** 🌱

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)