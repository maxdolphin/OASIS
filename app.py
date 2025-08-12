"""
Streamlit Web Application for Adaptive Organization Analysis

This web app provides an interactive interface for analyzing organizational 
sustainability using Ulanowicz's ecosystem theory and regenerative economics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our analysis modules
import sys
sys.path.append('src')
from ulanowicz_calculator import UlanowiczCalculator
from visualizer import SustainabilityVisualizer

# Configure page
st.set_page_config(
    page_title="Adaptive Organization Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e7e34;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .status-viable {
        color: #28a745;
        font-weight: bold;
    }
    .status-unsustainable {
        color: #dc3545;
        font-weight: bold;
    }
    .status-moderate {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 Adaptive Organization Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        Analyze organizational sustainability using Ulanowicz's ecosystem theory 
        and regenerative economics principles
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Control Panel")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["📊 Upload Data", "🧪 Use Sample Data", "⚡ Generate Synthetic Data", "📚 Learn More", "🔬 Formulas Reference"]
    )
    
    if analysis_mode == "📊 Upload Data":
        upload_data_interface()
    elif analysis_mode == "🧪 Use Sample Data":
        sample_data_interface()
    elif analysis_mode == "⚡ Generate Synthetic Data":
        synthetic_data_interface()
    elif analysis_mode == "📚 Learn More":
        learn_more_interface()
    elif analysis_mode == "🔬 Formulas Reference":
        formulas_reference_interface()

def upload_data_interface():
    """Interface for uploading custom data."""
    
    st.header("📊 Upload Your Organizational Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Supported Formats
        - **JSON**: Flow matrix with node names
        - **CSV**: Square matrix (with or without headers)
        
        ### Expected Structure
        Your data should represent communication flows between departments/teams.
        Values can be emails per month, document exchanges, or any flow metric.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['json', 'csv'],
            help="Upload a JSON or CSV file containing your organizational flow data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    if 'flows' in data and 'nodes' in data:
                        flow_matrix = np.array(data['flows'])
                        node_names = data['nodes']
                        org_name = data.get('organization', 'Your Organization')
                    else:
                        st.error("JSON file must contain 'flows' and 'nodes' keys")
                        return
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, index_col=0)
                    flow_matrix = df.values
                    node_names = df.columns.tolist()
                    org_name = uploaded_file.name.replace('.csv', '').replace('_', ' ').title()
                
                st.success(f"✅ Data loaded successfully! Found {len(node_names)} departments/teams")
                
                # Show preview
                st.subheader("📋 Data Preview")
                preview_df = pd.DataFrame(flow_matrix, index=node_names, columns=node_names)
                st.dataframe(preview_df.round(2))
                
                # Run analysis
                run_analysis(flow_matrix, node_names, org_name)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.markdown("""
        ### 📝 Data Format Example (JSON)
        ```json
        {
          "organization": "My Company",
          "nodes": ["Sales", "Marketing", "IT", "HR"],
          "flows": [
            [0.0, 8.0, 3.0, 2.0],
            [6.0, 0.0, 2.0, 1.0],
            [4.0, 5.0, 0.0, 3.0],
            [3.0, 2.0, 4.0, 0.0]
          ]
        }
        ```
        
        ### 📋 CSV Format
        ```
        ,Sales,Marketing,IT,HR
        Sales,0.0,8.0,3.0,2.0
        Marketing,6.0,0.0,2.0,1.0
        IT,4.0,5.0,0.0,3.0
        HR,3.0,2.0,4.0,0.0
        ```
        """)

def sample_data_interface():
    """Interface for using pre-built sample data."""
    
    st.header("🧪 Analyze Sample Organizations")
    
    # Load available sample datasets
    sample_datasets = {
        "TechFlow Innovations (Combined Flows)": "data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json",
        "TechFlow Innovations (Email Only)": "data/synthetic_organizations/email_flows/tech_company_email_matrix.json", 
        "TechFlow Innovations (Documents Only)": "data/synthetic_organizations/document_flows/tech_company_document_matrix.json",
        "Balanced Test Organization": "data/synthetic_organizations/combined_flows/balanced_org_test.json"
    }
    
    selected_dataset = st.selectbox("Choose a sample organization:", list(sample_datasets.keys()))
    
    if st.button("🚀 Analyze Selected Organization"):
        dataset_path = sample_datasets[selected_dataset]
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            flow_matrix = np.array(data['flows'])
            node_names = data['nodes']
            org_name = data.get('organization', selected_dataset)
            
            st.success(f"✅ Loaded {org_name}")
            
            # Show dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Departments", len(node_names))
            with col2:
                st.metric("Total Flows", int(np.sum(flow_matrix)))
            with col3:
                st.metric("Connections", int(np.count_nonzero(flow_matrix)))
            
            # Run analysis
            run_analysis(flow_matrix, node_names, org_name)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

def synthetic_data_interface():
    """Interface for generating synthetic data."""
    
    st.header("⚡ Generate Synthetic Organizational Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Organization Configuration")
        
        org_name = st.text_input("Organization Name", "My Test Company")
        org_type = st.selectbox("Organization Type", 
                               ["Technology Company", "Manufacturing", "Healthcare", "Finance", "Education", "Consulting"])
        
        # Department selection
        st.subheader("🏬 Departments")
        default_departments = ["Executive", "Product", "Engineering", "Sales", "Marketing", "Operations", "HR", "Finance"]
        
        departments = st.multiselect(
            "Select departments (minimum 3):",
            ["Executive", "Product", "Engineering", "Data_Science", "Sales", "Marketing", 
             "Customer_Success", "Operations", "HR", "Finance", "Legal", "R&D"],
            default=default_departments
        )
        
        if len(departments) < 3:
            st.warning("Please select at least 3 departments for meaningful analysis")
            return
    
    with col2:
        st.subheader("⚙️ Communication Parameters")
        
        communication_intensity = st.select_slider(
            "Communication Intensity:",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Higher intensity means more emails/communications between departments"
        )
        
        formality_level = st.select_slider(
            "Document Formality Level:",
            options=["Low", "Medium", "High"], 
            value="Medium",
            help="Higher formality means more formal documents and procedures"
        )
        
        organization_age = st.slider(
            "Organization Age (years):",
            min_value=1, max_value=50, value=5,
            help="Older organizations tend to have more established communication patterns and formal procedures"
        )
        
        st.subheader("🎲 Randomization")
        use_random_seed = st.checkbox("Use random seed for reproducibility", value=True)
        if use_random_seed:
            random_seed = st.number_input("Random seed:", min_value=1, max_value=1000, value=42)
        else:
            random_seed = None
    
    if st.button("🎯 Generate & Analyze Organization"):
        if len(departments) >= 3:
            with st.spinner("Generating synthetic organizational data..."):
                flow_matrix, node_names = generate_synthetic_organization(
                    departments, communication_intensity.lower(), formality_level.lower(), organization_age, random_seed
                )
                
                st.success(f"✅ Generated synthetic data for {org_name}")
                
                # Show generated data preview
                st.subheader("📊 Generated Flow Matrix")
                preview_df = pd.DataFrame(flow_matrix, index=node_names, columns=node_names)
                st.dataframe(preview_df.round(2))
                
                # Run analysis
                run_analysis(flow_matrix, node_names, org_name)

def generate_synthetic_organization(departments, intensity, formality, age, seed):
    """Generate synthetic organizational data."""
    
    if seed:
        np.random.seed(seed)
    
    n_depts = len(departments)
    
    # Base parameters
    intensity_params = {
        "low": {"base": 15, "variance": 10},
        "medium": {"base": 30, "variance": 20}, 
        "high": {"base": 50, "variance": 30}
    }
    
    formality_params = {
        "low": {"base": 8, "variance": 5},
        "medium": {"base": 15, "variance": 8},
        "high": {"base": 25, "variance": 12}
    }
    
    email_params = intensity_params[intensity]
    doc_params = formality_params[formality]
    
    # Age effects: older organizations have more established patterns
    age_factor = 1 + (age - 1) * 0.1  # 10% increase per year beyond first year
    hierarchy_factor = min(2.0, 1 + age * 0.05)  # More hierarchical patterns with age
    
    # Adjust parameters based on age
    email_params = {
        "base": email_params["base"] * age_factor,
        "variance": email_params["variance"] * (1 / age_factor)  # Less variance in older orgs
    }
    
    doc_params = {
        "base": doc_params["base"] * age_factor * hierarchy_factor,
        "variance": doc_params["variance"] * (1 / age_factor)
    }
    
    # Generate email flows
    email_matrix = np.zeros((n_depts, n_depts))
    for i in range(n_depts):
        for j in range(n_depts):
            if i != j:
                flow = max(1, email_params["base"] + np.random.normal(0, email_params["variance"]))
                email_matrix[i, j] = round(flow, 1)
    
    # Generate document flows (lower volume)
    doc_matrix = np.zeros((n_depts, n_depts))  
    for i in range(n_depts):
        for j in range(n_depts):
            if i != j:
                flow = max(1, doc_params["base"] + np.random.normal(0, doc_params["variance"]))
                doc_matrix[i, j] = round(flow, 1)
    
    # Combine with weights (documents carry more weight)
    combined_matrix = (email_matrix * 0.6) + (doc_matrix * 1.4)
    
    return combined_matrix, departments

def run_analysis(flow_matrix, node_names, org_name):
    """Run the complete Ulanowicz analysis and display results."""
    
    st.header(f"📊 Analysis Results: {org_name}")
    
    # Calculate metrics
    with st.spinner("Calculating sustainability metrics..."):
        calculator = UlanowiczCalculator(flow_matrix, node_names)
        extended_metrics = calculator.get_extended_metrics()
        assessments = calculator.assess_regenerative_health()
    
    # Main metrics overview
    display_metrics_overview(extended_metrics, assessments)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Core Metrics", "🌱 Regenerative Indicators", "📊 Visualizations", "🔄 Network Analysis", "📋 Detailed Report"])
    
    with tab1:
        display_core_metrics(extended_metrics)
    
    with tab2:
        display_regenerative_metrics(extended_metrics, assessments)
    
    with tab3:
        display_visualizations(calculator, extended_metrics)
    
    with tab4:
        display_network_analysis(calculator, extended_metrics, flow_matrix, node_names)
    
    with tab5:
        display_detailed_report(calculator, extended_metrics, assessments, org_name)

def display_metrics_overview(metrics, assessments):
    """Display high-level metrics overview."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        efficiency = metrics['network_efficiency']
        efficiency_color = "🟢" if 0.2 <= efficiency <= 0.6 else "🟡" if efficiency < 0.2 else "🔴"
        st.metric("Network Efficiency", f"{efficiency:.3f}", f"{efficiency_color} {get_efficiency_status(efficiency)}")
    
    with col2:
        robustness = metrics['robustness']
        robustness_color = "🟢" if robustness > 0.25 else "🟡" if robustness > 0.15 else "🔴"
        st.metric("Robustness", f"{robustness:.3f}", f"{robustness_color} {get_robustness_status(robustness)}")
    
    with col3:
        viable = "YES" if metrics['is_viable'] else "NO"
        viable_color = "🟢" if metrics['is_viable'] else "🔴"
        st.metric("Viable System", viable, f"{viable_color}")
    
    with col4:
        regen_capacity = metrics['regenerative_capacity']
        regen_color = "🟢" if regen_capacity > 0.2 else "🟡" if regen_capacity > 0.1 else "🔴"
        st.metric("Regenerative Capacity", f"{regen_capacity:.3f}", f"{regen_color}")
    
    # Overall assessment
    st.subheader("🎯 Overall System Health")
    sustainability_status = assessments['sustainability']
    
    if "VIABLE" in sustainability_status:
        st.success(f"✅ {sustainability_status}")
    elif "MODERATE" in sustainability_status or "GOOD" in sustainability_status:
        st.warning(f"⚠️ {sustainability_status}")
    else:
        st.error(f"❌ {sustainability_status}")

def display_core_metrics(metrics):
    """Display core Ulanowicz metrics."""
    
    st.subheader("📈 Core Ulanowicz Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### System Activity")
        st.metric("Total System Throughput (TST)", f"{metrics['total_system_throughput']:.1f}")
        st.metric("Average Mutual Information (AMI)", f"{metrics['average_mutual_information']:.3f}")
        st.metric("Ascendency (A)", f"{metrics['ascendency']:.1f}")
        
    with col2:
        st.markdown("### System Capacity")
        st.metric("Development Capacity (C)", f"{metrics['development_capacity']:.1f}")  
        st.metric("Overhead (Φ)", f"{metrics['overhead']:.1f}")
        
        # Ratios
        st.markdown("### Key Ratios")
        st.metric("Ascendency Ratio (A/C)", f"{metrics['ascendency_ratio']:.3f}")
        st.metric("Overhead Ratio (Φ/C)", f"{metrics['overhead_ratio']:.3f}")
    
    # Window of Viability
    st.subheader("🎯 Window of Viability")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lower Bound", f"{metrics['viability_lower_bound']:.1f}")
    with col2:
        st.metric("Upper Bound", f"{metrics['viability_upper_bound']:.1f}")
    with col3:
        current_pos = metrics['ascendency']
        lower = metrics['viability_lower_bound']
        upper = metrics['viability_upper_bound']
        
        if lower <= current_pos <= upper:
            st.success(f"✅ Current: {current_pos:.1f}")
        else:
            st.error(f"❌ Current: {current_pos:.1f}")

def display_regenerative_metrics(metrics, assessments):
    """Display regenerative economics indicators."""
    
    st.subheader("🌱 Regenerative Economics Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Flow & Structure")
        st.metric("Flow Diversity (H)", f"{metrics['flow_diversity']:.3f}")
        st.metric("Structural Information (SI)", f"{metrics['structural_information']:.3f}")
        st.metric("Effective Link Density", f"{metrics['effective_link_density']:.3f}")
        st.metric("Trophic Depth", f"{metrics['trophic_depth']:.3f}")
    
    with col2:
        st.markdown("### System Dynamics")  
        st.metric("Robustness (R)", f"{metrics['robustness']:.3f}")
        st.metric("Redundancy", f"{metrics['redundancy']:.3f}")
        st.metric("Network Efficiency", f"{metrics['network_efficiency']:.3f}")
        st.metric("Regenerative Capacity", f"{metrics['regenerative_capacity']:.3f}")
    
    # Health assessments
    st.subheader("🏥 Health Assessment Breakdown")
    
    assessment_colors = {
        'HIGH': '🟢', 'GOOD': '🟢', 'OPTIMAL': '🟢',
        'MODERATE': '🟡', 'VIABLE': '🟡', 
        'LOW': '🔴', 'UNSUSTAINABLE': '🔴', 'WEAK': '🔴'
    }
    
    for category, assessment in assessments.items():
        status = assessment.split(' - ')[0]
        color = assessment_colors.get(status, '⚪')
        st.write(f"{color} **{category.title()}**: {assessment}")

def display_visualizations(calculator, metrics):
    """Display interactive visualizations."""
    
    st.subheader("📊 Interactive Visualizations")
    
    # Create visualizer
    visualizer = SustainabilityVisualizer(calculator)
    
    # Robustness curve
    st.subheader("💪 Robustness Analysis")
    robustness_fig = create_robustness_curve(metrics)
    st.plotly_chart(robustness_fig, use_container_width=True)
    
    # Network flow heatmap
    st.subheader("🔥 Network Flow Heatmap")
    flow_fig = create_flow_heatmap(calculator.flow_matrix, calculator.node_names)
    st.plotly_chart(flow_fig, use_container_width=True)


def create_robustness_curve(metrics):
    """Create robustness curve visualization."""
    
    efficiency_range = np.linspace(0.01, 0.99, 100)
    development_capacity = metrics['development_capacity']
    
    # Create normalized robustness curve (shape only, not absolute values)
    normalized_robustness = []
    for eff in efficiency_range:
        # Normalized robustness function (without log(C) scaling)
        robustness_shape = eff * (1 - eff)
        normalized_robustness.append(robustness_shape)
    
    # Scale the curve to make it visible relative to current organization
    max_shape = max(normalized_robustness)
    current_efficiency = metrics['network_efficiency'] 
    current_robustness = metrics['robustness']
    
    # Scale curve so current organization's theoretical position matches actual
    if current_efficiency > 0 and max_shape > 0:
        theoretical_shape = current_efficiency * (1 - current_efficiency)
        scale_factor = current_robustness / theoretical_shape if theoretical_shape > 0 else 1
    else:
        scale_factor = 1
    
    scaled_robustness = [r * scale_factor for r in normalized_robustness]
    
    fig = go.Figure()
    
    # Robustness curve (normalized and scaled)
    fig.add_trace(go.Scatter(x=efficiency_range, y=scaled_robustness, mode='lines',
                            name='Theoretical Robustness Curve', line=dict(width=3, color='blue', dash='dot')))
    
    # Current organization position (actual calculated robustness)
    fig.add_trace(go.Scatter(x=[current_efficiency], y=[current_robustness], mode='markers',
                            marker=dict(size=15, color='red'), name='Your Organization',
                            hovertemplate='Your Position<br>Efficiency: %{x:.3f}<br>Robustness: %{y:.3f}<extra></extra>'))
    
    # Mathematical peak (50% efficiency - theoretical maximum)
    math_optimal_efficiency = 0.5
    math_optimal_robustness = math_optimal_efficiency * (1 - math_optimal_efficiency) * scale_factor
    fig.add_trace(go.Scatter(x=[math_optimal_efficiency], y=[math_optimal_robustness], mode='markers',
                            marker=dict(size=12, color='green', symbol='star'), name='Mathematical Peak',
                            hovertemplate='Theoretical Maximum<br>Efficiency: %{x:.3f}<br>Peak Robustness: %{y:.3f}<extra></extra>'))
    
    # Empirical optimum (37% efficiency - Ulanowicz's research finding)
    empirical_optimal_efficiency = 0.37
    empirical_optimal_robustness = empirical_optimal_efficiency * (1 - empirical_optimal_efficiency) * scale_factor
    fig.add_trace(go.Scatter(x=[empirical_optimal_efficiency], y=[empirical_optimal_robustness], mode='markers',
                            marker=dict(size=10, color='orange', symbol='diamond'), name='Empirical Optimum',
                            hovertemplate='Ulanowicz Optimum<br>Efficiency: %{x:.3f}<br>Real-world Peak: %{y:.3f}<extra></extra>'))
    
    # Add viability bounds
    fig.add_vrect(x0=0.2, x1=0.6, fillcolor="green", opacity=0.1, 
                  annotation_text="Window of Viability", annotation_position="top left")
    
    # Add annotations
    fig.add_annotation(
        x=current_efficiency, y=current_robustness,
        text=f"Your Org<br>α={current_efficiency:.3f}",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red",
        xshift=20, yshift=20
    )
    
    fig.update_layout(
        title='System Robustness vs Network Efficiency<br><sub>Your Organization\'s Position Relative to Theoretical Optimum</sub>',
        xaxis_title='Network Efficiency (α = A/C) - Relative Ascendency',
        yaxis_title='Robustness - Ability to Handle Disturbances',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_flow_heatmap(flow_matrix, node_names):
    """Create network flow heatmap."""
    
    fig = go.Figure(data=go.Heatmap(
        z=flow_matrix,
        x=node_names,
        y=node_names,
        colorscale='Viridis',
        colorbar=dict(title="Flow Intensity")
    ))
    
    fig.update_layout(title='Network Flow Matrix',
                      xaxis_title='To Department', yaxis_title='From Department')
    
    return fig


def display_network_analysis(calculator, metrics, flow_matrix, node_names):
    """Display network analysis details."""
    
    st.subheader("🔗 Network Structure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Network Properties")
        st.write(f"**Nodes**: {calculator.n_nodes}")
        st.write(f"**Total Connections**: {np.count_nonzero(flow_matrix)}")
        st.write(f"**Network Density**: {np.count_nonzero(flow_matrix) / (calculator.n_nodes ** 2):.3f}")
        st.write(f"**Total Flow**: {np.sum(flow_matrix):.1f}")
        st.write(f"**Average Flow**: {np.mean(flow_matrix[flow_matrix > 0]):.1f}")
    
    with col2:
        st.markdown("### Flow Statistics")
        st.write(f"**Max Flow**: {np.max(flow_matrix):.1f}")
        st.write(f"**Min Flow**: {np.min(flow_matrix[flow_matrix > 0]):.1f}")
        st.write(f"**Flow Std Dev**: {np.std(flow_matrix[flow_matrix > 0]):.1f}")
    
    # Department analysis
    st.subheader("🏬 Department Analysis")
    
    # Calculate department metrics
    outflows = np.sum(flow_matrix, axis=1)
    inflows = np.sum(flow_matrix, axis=0)
    
    dept_df = pd.DataFrame({
        'Department': node_names,
        'Outflow': outflows,
        'Inflow': inflows,
        'Total Flow': outflows + inflows,
        'Flow Balance': outflows - inflows
    })
    
    dept_df = dept_df.sort_values('Total Flow', ascending=False)
    st.dataframe(dept_df)
    
    # Top connections
    st.subheader("🔝 Strongest Connections")
    connections = []
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i != j and flow_matrix[i, j] > 0:
                connections.append({
                    'From': node_names[i],
                    'To': node_names[j], 
                    'Flow': flow_matrix[i, j]
                })
    
    connections_df = pd.DataFrame(connections).sort_values('Flow', ascending=False).head(10)
    st.dataframe(connections_df)

def display_detailed_report(calculator, metrics, assessments, org_name):
    """Display detailed analysis report."""
    
    st.subheader("📋 Comprehensive Analysis Report")
    
    # Generate report text
    report = generate_text_report(calculator, metrics, assessments, org_name)
    
    # Display report
    st.text_area("Full Report", report, height=600)
    
    # Download options
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download Report as Text"):
            st.download_button(
                label="Download TXT",
                data=report,
                file_name=f"{org_name.replace(' ', '_')}_analysis_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Download Data as JSON"):
            data_export = {
                'organization': org_name,
                'metrics': metrics,
                'assessments': assessments,
                'flow_matrix': calculator.flow_matrix.tolist(),
                'node_names': calculator.node_names
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(data_export, indent=2),
                file_name=f"{org_name.replace(' ', '_')}_analysis_data.json",
                mime="application/json"
            )

def generate_text_report(calculator, metrics, assessments, org_name):
    """Generate comprehensive text report."""
    
    report = f"""
ADAPTIVE ORGANIZATION ANALYSIS REPORT
=====================================

Organization: {org_name}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: Ulanowicz-Fath Regenerative Economics Framework

EXECUTIVE SUMMARY
=================
Sustainability Status: {assessments['sustainability']}
Overall System Health: {'HEALTHY' if metrics['is_viable'] and metrics['robustness'] > 0.15 else 'NEEDS ATTENTION'}

CORE ULANOWICZ METRICS
=====================
Total System Throughput (TST): {metrics['total_system_throughput']:.3f}
Average Mutual Information (AMI): {metrics['average_mutual_information']:.3f}
Ascendency (A): {metrics['ascendency']:.3f}
Development Capacity (C): {metrics['development_capacity']:.3f}
Overhead (Φ): {metrics['overhead']:.3f}

EXTENDED REGENERATIVE METRICS
============================
Flow Diversity (H): {metrics['flow_diversity']:.3f}
Structural Information (SI): {metrics['structural_information']:.3f}
Robustness (R): {metrics['robustness']:.3f}
Network Efficiency: {metrics['network_efficiency']:.3f}
Regenerative Capacity: {metrics['regenerative_capacity']:.3f}

SYSTEM RATIOS
=============
Ascendency Ratio (A/C): {metrics['ascendency_ratio']:.3f}
Overhead Ratio (Φ/C): {metrics['overhead_ratio']:.3f}
Redundancy: {metrics['redundancy']:.3f}

WINDOW OF VIABILITY
==================
Lower Bound: {metrics['viability_lower_bound']:.3f}
Upper Bound: {metrics['viability_upper_bound']:.3f}
Current Position: {metrics['ascendency']:.3f}
Is Viable: {'YES' if metrics['is_viable'] else 'NO'}

HEALTH ASSESSMENT
================
"""
    
    for category, assessment in assessments.items():
        report += f"{category.title()}: {assessment}\n"
    
    report += f"""
NETWORK PROPERTIES
==================
Nodes: {calculator.n_nodes}
Total Connections: {np.count_nonzero(calculator.flow_matrix)}
Network Density: {np.count_nonzero(calculator.flow_matrix) / (calculator.n_nodes ** 2):.3f}
Effective Link Density: {metrics['effective_link_density']:.3f}
Trophic Depth: {metrics['trophic_depth']:.3f}

RECOMMENDATIONS
===============
"""
    
    # Add recommendations based on metrics
    if metrics['network_efficiency'] < 0.2:
        report += "• System efficiency is low - consider streamlining processes and improving coordination\n"
    elif metrics['network_efficiency'] > 0.6:
        report += "• System may be over-optimized - consider adding redundancy for resilience\n"
    
    if metrics['robustness'] < 0.15:
        report += "• System lacks robustness - focus on building adaptive capacity and resilience\n"
    elif metrics['robustness'] > 0.25:
        report += "• System demonstrates strong robustness - maintain current balance\n"
    
    if not metrics['is_viable']:
        report += "• System is outside window of viability - significant restructuring may be needed\n"
    
    return report

def get_efficiency_status(efficiency):
    """Get efficiency status string."""
    if 0.2 <= efficiency <= 0.6:
        return "Optimal"
    elif efficiency < 0.2:
        return "Low"
    else:
        return "High"

def get_robustness_status(robustness):
    """Get robustness status string."""
    if robustness > 0.25:
        return "High"
    elif robustness > 0.15:
        return "Moderate"
    else:
        return "Low"

def learn_more_interface():
    """Educational interface about the methodology."""
    
    st.header("📚 Learn About Regenerative Economics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Overview", "📊 Key Metrics", "🎯 Window of Viability", "🔬 Research"])
    
    with tab1:
        st.markdown("""
        ## 🌱 Regenerative Economics Framework
        
        This analysis system is based on **Robert Ulanowicz's Ecosystem Theory** extended with 
        **Regenerative Economics** principles developed by Brian Fath and Ulanowicz.
        
        ### Core Concept
        
        Organizations, like ecosystems, need to balance **efficiency** (organization) with **resilience** 
        (flexibility) to remain sustainable. Too much efficiency leads to brittleness, while too 
        little leads to chaos.
        
        ### Key Insights
        
        - **Flow Networks**: Organizations are networks of information, resource, and communication flows
        - **Balance is Critical**: Optimal systems operate in a "window of viability"
        - **Regenerative Capacity**: Healthy systems can adapt and self-renew
        - **Measurable Sustainability**: Network properties predict long-term viability
        """)
    
    with tab2:
        st.markdown("""
        ## 📊 Key Metrics Explained
        
        ### Core Ulanowicz Metrics
        
        **Total System Throughput (TST)**
        - Total activity/flow in the organization
        - Higher values indicate more active systems
        
        **Average Mutual Information (AMI)**
        - Measures organization and constraint in flows
        - Higher values indicate more structured communication
        
        **Ascendency (A = TST × AMI)**
        - System "power" - combination of size and organization
        - Represents current organizational capacity
        
        **Development Capacity (C)**
        - Theoretical maximum ascendency possible
        - Represents total system potential
        
        **Overhead (Φ = C - A)**
        - System flexibility and redundancy
        - Provides resilience and adaptive capacity
        
        ### Extended Regenerative Metrics
        
        **Robustness (R)**
        - System's ability to maintain function under stress
        - Balances efficiency with resilience
        - Formula: R = (A/C) × (1 - A/C) × log(C)
        
        **Flow Diversity (H)**
        - Evenness of communication distribution
        - Higher diversity indicates more distributed systems
        
        **Regenerative Capacity**
        - System's potential for self-renewal and adaptation
        - Combines robustness with optimization distance
        """)
    
    with tab3:
        st.markdown("""
        ## 🎯 Window of Viability
        
        The **Window of Viability** defines the sustainable operating range for organizations.
        
        ### Boundaries
        
        **Lower Bound (~20% of Development Capacity)**
        - Below this: System too chaotic, lacks organization
        - Results in: Inefficiency, confusion, poor coordination
        
        **Upper Bound (~60% of Development Capacity)**
        - Above this: System too rigid, over-organized
        - Results in: Brittleness, inability to adapt, stagnation
        
        **Optimal Range (20-60%)**
        - Sweet spot for sustainable operations
        - Balances efficiency with adaptability
        - Enables both performance and resilience
        
        ### Theoretical Optimum
        
        **Maximum Robustness** occurs around **37% efficiency ratio**
        - Best balance of organization and flexibility
        - Highest adaptive capacity
        - Most sustainable long-term position
        """)
    
    with tab4:
        st.markdown("""
        ## 🔬 Research Foundation
        
        ### Primary Sources
        
        **Robert E. Ulanowicz**
        - "A Third Window: Natural Life Beyond Newton and Darwin"
        - Pioneer of ecosystem network analysis
        - Developer of ascendency concept
        
        **Fath & Ulanowicz (2019)**
        - "Measuring Regenerative Economics: 10 principles and measures undergirding systemic economic health"
        - Extended framework to economic systems
        - Added regenerative capacity concepts
        
        ### Key Research Areas
        
        **Network Ecology**
        - Flow analysis in complex systems
        - Information theory applications
        - Sustainability metrics development
        
        **Systems Theory**
        - Complex adaptive systems
        - Resilience and robustness
        - Self-organization principles
        
        **Regenerative Economics**
        - Alternative to traditional growth models
        - Focus on long-term sustainability
        - Integration of ecological principles
        
        ### Applications
        
        - Organizational design and restructuring
        - Supply chain optimization
        - Urban planning and development
        - Economic policy analysis
        - Ecosystem management
        """)

def formulas_reference_interface():
    """Complete formulas reference for all indicators."""
    
    st.header("🔬 Complete Formulas Reference")
    st.markdown("""
    This page contains all mathematical formulations used in the Adaptive Organization Analysis system,
    organized by category and based on peer-reviewed scientific literature.
    """)
    
    # Create tabs for different categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧮 Core Ulanowicz IT", "🌱 Regenerative Economics", "📊 Network Analysis", 
        "🎯 Sustainability Metrics", "⚙️ Extended Indicators"
    ])
    
    with tab1:
        st.subheader("🧮 Core Information Theory Formulations")
        st.markdown("*Based on Ulanowicz et al. (2009) - Foundational paper*")
        
        st.markdown("""
        ### **Total System Throughput (TST)**
        ```
        TST = Σ T_ij
        ```
        Where T_ij is the flow from compartment i to compartment j
        
        ### **Development Capacity (C)** 
        ```
        C = -Σ(T_ij × log(T_ij / T··))
        ```
        - **Equation (11)** from Ulanowicz et al. (2009)
        - Represents scaled system indeterminacy
        - Units: flow-bits
        
        ### **Ascendency (A)**
        ```
        A = Σ(T_ij × log(T_ij × T·· / (T_i· × T_·j)))
        ```
        - **Equation (12)** from Ulanowicz et al. (2009)
        - Scaled mutual constraint (organized power)
        - Units: flow-bits
        
        ### **Reserve (Φ)**
        ```
        Φ = C - A
        ```
        - **Equation (14)** from Ulanowicz et al. (2009)
        - System flexibility and reserve capacity
        - Units: flow-bits
        
        ### **Relative Ascendency (α)**
        ```
        α = A / C
        ```
        - **Key sustainability metric**
        - Dimensionless ratio (0 to 1)
        - Optimal range: 0.2 - 0.6
        
        ### **Fundamental Relationship**
        ```
        C = A + Φ
        ```
        - **Mathematical constraint** from Information Theory
        - Used for validation (should hold exactly)
        """)
    
    with tab2:
        st.subheader("🌱 Regenerative Economics Formulations")
        st.markdown("*Based on Fath & Ulanowicz (2019) and Goerner et al. (2009)*")
        
        st.markdown("""
        ### **Robustness (R)**
        ```
        R = (A/C) × (1 - A/C) × log(C)
        ```
        - **Fath-Ulanowicz formulation**
        - Balances efficiency with resilience
        - Peak occurs around α = 0.37 (empirically)
        
        ### **Flow Diversity (H)**
        ```
        H = -Σ(p_ij × log(p_ij))
        where p_ij = T_ij / TST
        ```
        - **Shannon entropy** of flow distribution
        - Higher values = more evenly distributed flows
        
        ### **Structural Information (SI)**
        ```
        SI = log(n²) - H
        ```
        - Network constraint independent of magnitudes
        - n = number of nodes
        
        ### **Regenerative Capacity**
        ```
        RC = R × (1 - |α - α_opt|)
        where α_opt = 0.37
        ```
        - Combines robustness with distance from optimum
        - Measures self-renewal potential
        """)
    
    with tab3:
        st.subheader("📊 Network Analysis Formulations")
        
        st.markdown("""
        ### **Network Efficiency**
        ```
        Efficiency = A / C = α
        ```
        - Same as relative ascendency
        - Measures organizational constraint
        
        ### **Redundancy**
        ```
        Redundancy = Φ / C = 1 - α
        ```
        - Alternative pathways and backup capacity
        - Complement of efficiency
        
        ### **Average Mutual Information (AMI)**
        ```
        AMI = Σ(T_ij × log(T_ij × TST / (T_i· × T_·j))) / TST
        ```
        - Degree of organization in flow patterns
        - Higher values = more structured
        
        ### **Effective Link Density**
        ```
        ELD = (L_active / L_max) × (AMI / AMI_max)
        ```
        - L_active = number of non-zero flows
        - L_max = n²
        - Weighted by information content
        
        ### **Trophic Depth**
        ```
        TD = Average shortest path length (weighted)
        ```
        - Calculated using NetworkX algorithms
        - Indicates hierarchical organization
        """)
    
    with tab4:
        st.subheader("🎯 Sustainability Assessment Formulations")
        
        st.markdown("""
        ### **Window of Viability**
        ```
        Lower Bound = 0.2 × C
        Upper Bound = 0.6 × C
        Viable = Lower Bound ≤ A ≤ Upper Bound
        ```
        - **Empirical bounds** from Ulanowicz research
        - Based on natural ecosystem observations
        
        ### **Sustainability Classification**
        ```
        if α < 0.2:  "Too chaotic (low organization)"
        if α > 0.6:  "Too rigid (over-organized)" 
        if 0.2 ≤ α ≤ 0.6:  "Viable system"
        ```
        
        ### **Optimal Robustness Point**
        ```
        Mathematical Peak: α = 0.5 (derivative = 0)
        Empirical Optimum: α = 0.37 (Ulanowicz research)
        ```
        
        ### **Health Assessment Logic**
        ```
        Robustness: HIGH (>0.25), MODERATE (0.15-0.25), LOW (<0.15)
        Efficiency: OPTIMAL (0.2-0.6), LOW (<0.2), HIGH (>0.6)
        Resilience: Based on redundancy and diversity thresholds
        ```
        """)
    
    with tab5:
        st.subheader("⚙️ Extended Indicator Formulations")
        
        st.markdown("""
        ### **Input/Output Throughput**
        ```
        T_i· = Σ_j T_ij  (output from node i)
        T_·j = Σ_i T_ij  (input to node j)
        ```
        
        ### **Total Throughput per Node**
        ```
        TT_k = T_k· + T_·k
        ```
        - Sum of all flows through node k
        
        ### **Flow Balance**
        ```
        Balance_k = T_k· - T_·k
        ```
        - Positive = net outflow, Negative = net inflow
        
        ### **Network Density**
        ```
        Density = L_active / L_possible
        where L_possible = n × (n-1)
        ```
        - Fraction of possible connections actually used
        
        ### **Validation Metrics**
        ```
        Fundamental Error = |C - (A + Φ)| / C
        Valid = Error < 0.001 (0.1% tolerance)
        ```
        - Mathematical consistency check
        """)
    
    # Mathematical notation guide
    st.markdown("---")
    st.subheader("📝 Notation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Flow Variables:**
        - `T_ij` = Flow from node i to node j
        - `T_i·` = Total outflow from node i
        - `T_·j` = Total inflow to node j  
        - `T··` = Total system throughput (TST)
        - `n` = Number of nodes/compartments
        """)
    
    with col2:
        st.markdown("""
        **Information Theory:**
        - `log` = Natural logarithm (ln)
        - `Σ` = Summation over all flows
        - `α` = Alpha (relative ascendency)
        - `Φ` = Phi (reserve/overhead)
        - Units: "flow-bits" for information measures
        """)

def show_app_version():
    """Display app version information."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        <strong>Adaptive Organization Analysis System</strong><br>
        Version 2.0.0 - Information Theory Corrected<br>
        Built with Ulanowicz-Fath Regenerative Economics Framework<br>
        <em>Phase 1 Complete: Core IT Formulations Implemented</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_app_version()