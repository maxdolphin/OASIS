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
from network_generator import OrganizationalNetworkGenerator, NETWORK_TYPES

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
        color: #f59e0b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    # Show different pages based on current state
    if st.session_state.current_page == 'analysis':
        show_analysis_page()
    else:
        show_main_page()

def show_main_page():
    """Show the main interface page."""
    
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
        ["📊 Upload Data", "🧪 Use Sample Data", "⚡ Generate Synthetic Data", "📚 Learn More", "🌱 10 Principles", "🔬 Formulas Reference"]
    )
    
    if analysis_mode == "📊 Upload Data":
        upload_data_interface()
    elif analysis_mode == "🧪 Use Sample Data":
        sample_data_interface()
    elif analysis_mode == "⚡ Generate Synthetic Data":
        synthetic_data_interface()
    elif analysis_mode == "📚 Learn More":
        learn_more_interface()
    elif analysis_mode == "🌱 10 Principles":
        ten_principles_interface()
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
                
                # Run analysis button
                if st.button("🚀 Run Analysis", type="primary"):
                    # Store data in session state and navigate to analysis page
                    st.session_state.analysis_data = {
                        'flow_matrix': flow_matrix,
                        'node_names': node_names,
                        'org_name': org_name,
                        'source': 'uploaded'
                    }
                    st.session_state.current_page = 'analysis'
                    st.rerun()
                
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
    """Interface for using built-in and user-saved sample data."""
    
    st.header("🧪 Analyze Sample Organizations & Ecosystems")
    st.markdown("Choose from organizational samples, real ecosystems from scientific literature, or your saved networks.")
    
    # Load all available datasets (built-in + user-saved + ecosystems)
    all_datasets = load_all_sample_datasets()
    
    if not all_datasets:
        st.warning("No sample datasets available. Try generating some networks first!")
        return
    
    # Organize datasets by type for better UX
    builtin_datasets = {k: v for k, v in all_datasets.items() if v["type"] == "builtin"}
    ecosystem_datasets = {k: v for k, v in all_datasets.items() if v["type"] == "ecosystem"}
    user_datasets = {k: v for k, v in all_datasets.items() if v["type"] == "user_saved"}
    
    # Show counts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📁 **Samples**: {len(builtin_datasets)}")
    with col2:
        st.info(f"🌿 **Ecosystems**: {len(ecosystem_datasets)}")
    with col3:
        st.info(f"💾 **Your Networks**: {len(user_datasets)}")
    
    # Dataset selection
    selected_dataset = st.selectbox("Choose an organization:", list(all_datasets.keys()))
    dataset_info = all_datasets[selected_dataset]
    
    # Show metadata based on dataset type
    if dataset_info["type"] == "ecosystem" and "metadata" in dataset_info:
        metadata = dataset_info["metadata"]
        
        with st.expander("🌿 Ecosystem Details", expanded=True):
            st.write(f"**Source**: {metadata.get('source', 'N/A')}")
            st.write(f"**Description**: {metadata.get('description', 'N/A')}")
            st.write(f"**Units**: {metadata.get('units', 'N/A')}")
            
            # Show published metrics if available
            published = metadata.get('published_metrics', {})
            if published:
                st.subheader("📊 Published Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'total_system_throughput' in published:
                        st.metric("TST", f"{published['total_system_throughput']:.0f}")
                with col2:
                    if 'ascendency' in published:
                        st.metric("Ascendency", f"{published['ascendency']:.0f}")
                with col3:
                    if 'ascendency_ratio' in published:
                        st.metric("A/C Ratio", f"{published['ascendency_ratio']:.3f}")
                if 'note' in published:
                    st.info(f"📝 {published['note']}")
    
    elif dataset_info["type"] == "user_saved" and "metadata" in dataset_info:
        metadata = dataset_info["metadata"]
        
        with st.expander("📋 Network Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Type**: {metadata.get('network_description', 'N/A')}")
                st.write(f"**Nodes**: {metadata.get('actual_nodes', 'N/A')}")
                st.write(f"**Edges**: {metadata.get('actual_edges', 'N/A')}")
            with col2:
                st.write(f"**Density**: {metadata.get('actual_density', 0):.4f}")
                st.write(f"**Total Flow**: {metadata.get('total_flow', 0):.1f}")
                st.write(f"**Hub Amplification**: {metadata.get('hub_amplification', 'N/A')}")
            with col3:
                created = metadata.get('created', 'Unknown')
                if created != 'Unknown':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        created = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                st.write(f"**Created**: {created}")
                st.write(f"**Flow Range**: {metadata.get('flow_range', 'N/A')}")
    
    # Analysis buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("🚀 Analyze Selected Organization", type="primary")
    with col2:
        if dataset_info["type"] == "user_saved":
            if st.button("🗑️ Delete This Network", type="secondary"):
                try:
                    import os
                    os.remove(dataset_info["path"])
                    st.success("✅ Network deleted successfully!")
                    st.rerun()  # Refresh the interface
                except Exception as e:
                    st.error(f"❌ Failed to delete: {str(e)}")
    
    if analyze_button:
        dataset_path = dataset_info["path"]
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            flow_matrix = np.array(data['flows'])
            node_names = data['nodes']
            org_name = data.get('organization', selected_dataset.split(' - ')[0])  # Clean up display name
            
            # Store data in session state and navigate to analysis page
            st.session_state.analysis_data = {
                'flow_matrix': flow_matrix,
                'node_names': node_names,
                'org_name': org_name,
                'source': 'sample_data'
            }
            st.session_state.current_page = 'analysis'
            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

def synthetic_data_interface():
    """Visual Network Generator Interface."""
    
    st.header("⚡ Visual Network Generator")
    st.markdown("""
    Create and analyze organizational networks by adjusting structure and flow patterns.
    See how different network topologies affect sustainability metrics in real-time.
    """)
    
    # Initialize session state for network data
    if 'generated_network' not in st.session_state:
        st.session_state.generated_network = None
    if 'flow_matrix' not in st.session_state:
        st.session_state.flow_matrix = None
    
    # Network Controls section
    st.subheader("🎛️ Network Controls")
    
    # Organization info
    org_name = st.text_input("Organization Name", "Generated Network Org")
    
    # Network structure selection
    st.markdown("### 🏗️ Network Structure")
    network_type = st.selectbox(
            "Network Type:",
            options=list(NETWORK_TYPES.keys()),
            format_func=lambda x: NETWORK_TYPES[x]["name"],
            help="Choose the organizational structure pattern"
    )
    
    # Show network description
    selected_type = NETWORK_TYPES[network_type]
    st.info(f"**{selected_type['description']}**\n\n"
            f"Characteristics: {selected_type['characteristics']}\n\n"
            f"Use cases: {selected_type['use_cases']}")
    
    # Size controls
    st.markdown("### 📏 Size Controls")
    num_nodes = st.slider("Number of Nodes:", 3, 1000, 10, 
                         help="Number of departments/units in the organization")
    
    # Adjust density range based on network size for performance
    if num_nodes > 100:
        max_density = min(0.3, 500 / (num_nodes * (num_nodes - 1)))
        density = st.slider("Network Density:", 0.01, max_density, min(0.1, max_density/2),
                           help="Density limited for large networks to ensure performance")
    else:
        density = st.slider("Network Density:", 0.1, 0.8, 0.3,
                           help="Fraction of possible connections that exist")
    
    # Flow controls
    st.markdown("### 💧 Flow Parameters")
    flow_range = st.slider("Flow Intensity Range:", 1, 100, (5, 50),
                          help="Minimum and maximum flow values")
    
    hub_amplification = st.slider("Hub Amplification:", 0.0, 2.0, 0.5,
                                 help="How much extra flow large hubs receive")
    
    # Additional parameters based on network type
    if network_type == 'small_world':
        rewiring_prob = st.slider("Rewiring Probability:", 0.1, 0.9, 0.3,
                                 help="Probability of rewiring edges for shortcuts")
    elif network_type == 'hierarchical':
        branching_factor = st.slider("Branching Factor:", 2, 5, 2,
                                    help="Number of subordinates per manager")
    elif network_type == 'community':
        num_communities = st.slider("Number of Communities:", 2, 6, 3,
                                   help="Number of distinct departments/groups")
    
    # Randomization
    st.markdown("### 🎲 Randomization")
    use_random_seed = st.checkbox("Use random seed", value=True)
    if use_random_seed:
        random_seed = st.number_input("Seed:", min_value=1, max_value=1000, value=42)
    else:
        random_seed = None
    
    # Save option
    save_to_samples = st.checkbox("💾 Save to Sample Data after generation", value=False,
                                  help="Save this network to your sample data collection for future use")
    
    # Performance warning for large networks
    if num_nodes > 500:
        st.warning("⚠️ **Large Network**: Networks with >500 nodes may take longer to generate and analyze.")
    elif num_nodes > 200:
        st.info("ℹ️ **Medium Network**: Visualization will be replaced with degree distribution chart.")
    
    # Generate button
    if st.button("🚀 Generate & Analyze Network", type="primary"):
        generation_time = "Generating network..."
        if num_nodes > 500:
            generation_time = "Generating large network... this may take a moment..."
            
        with st.spinner(generation_time):
            # Initialize generator
            generator = OrganizationalNetworkGenerator(seed=random_seed)
            
            # Additional kwargs based on network type
            kwargs = {}
            if network_type == 'small_world':
                kwargs['rewiring_prob'] = rewiring_prob
            elif network_type == 'hierarchical':
                kwargs['branching_factor'] = branching_factor
            elif network_type == 'community':
                kwargs['num_communities'] = num_communities
            
            # Generate network structure
            G = generator.generate_network(network_type, num_nodes, density, **kwargs)
            
            # Add flow weights
            G_weighted = generator.add_flow_weights(
                G, flow_range[0], flow_range[1], hub_amplification
            )
            
            # Generate flow matrix
            flow_matrix = generator.network_to_flow_matrix(G_weighted)
            node_names = [f"Unit_{i}" for i in range(flow_matrix.shape[0])]
            
            # Save to samples if requested
            if save_to_samples:
                save_network_to_samples(org_name, G_weighted, network_type, selected_type, 
                                      num_nodes, density, flow_range, hub_amplification)
                st.success("✅ Network saved to sample data!")
            
            # Store data in session state and navigate to analysis page
            st.session_state.analysis_data = {
                'flow_matrix': flow_matrix,
                'node_names': node_names,
                'org_name': org_name,
                'network': G_weighted,
                'source': 'synthetic'
            }
            st.session_state.current_page = 'analysis'
            
            st.success("✅ Network generated successfully! Navigating to analysis...")
            st.rerun()

def save_network_to_samples(org_name, network, network_type, selected_type, num_nodes, density, flow_range, hub_amplification):
    """Save generated network to user sample data collection."""
    
    try:
        from datetime import datetime
        import os
        
        # Create safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in org_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"{safe_name}_{timestamp}.json"
        
        # Convert network to flow matrix
        generator = OrganizationalNetworkGenerator()
        flow_matrix = generator.network_to_flow_matrix(network)
        node_names = [f"Unit_{i}" for i in range(network.number_of_nodes())]
        
        # Create metadata
        save_data = {
            "organization": org_name,
            "nodes": node_names,
            "flows": flow_matrix.tolist(),
            "metadata": {
                "created": datetime.now().isoformat(),
                "network_type": network_type,
                "network_description": selected_type["name"],
                "num_nodes": num_nodes,
                "density": density,
                "flow_range": flow_range,
                "hub_amplification": hub_amplification,
                "actual_nodes": network.number_of_nodes(),
                "actual_edges": network.number_of_edges(),
                "actual_density": network.number_of_edges() / (network.number_of_nodes() * (network.number_of_nodes() - 1)),
                "total_flow": float(np.sum(flow_matrix)),
                "saved_from": "Visual Network Generator"
            }
        }
        
        # Save to user directory
        save_path = f"data/user_saved_networks/{filename}"
        os.makedirs("data/user_saved_networks", exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        st.success(f"✅ **Network Saved!**\n\nSaved as: `{filename}`\n\nYou can now find it in the '🧪 Use Sample Data' section under 'User Saved Networks'.")
        
    except Exception as e:
        st.error(f"❌ Failed to save network: {str(e)}")

def load_all_sample_datasets():
    """Load both built-in and user-saved datasets."""
    
    datasets = {}
    
    # Built-in sample datasets
    builtin_datasets = {
        "TechFlow Innovations (Combined Flows)": "data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json",
        "TechFlow Innovations (Email Only)": "data/synthetic_organizations/email_flows/tech_company_email_matrix.json", 
        "TechFlow Innovations (Documents Only)": "data/synthetic_organizations/document_flows/tech_company_document_matrix.json",
        "Balanced Test Organization": "data/synthetic_organizations/combined_flows/balanced_org_test.json"
    }
    
    # Add built-in datasets
    import os
    for name, path in builtin_datasets.items():
        if os.path.exists(path):
            datasets[f"📁 {name}"] = {"path": path, "type": "builtin"}
    
    # Load ecosystem samples
    ecosystem_dir = "data/ecosystem_samples"
    if os.path.exists(ecosystem_dir):
        ecosystem_files = [f for f in os.listdir(ecosystem_dir) if f.endswith('.json')]
        for filename in ecosystem_files:
            filepath = os.path.join(ecosystem_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                org_name = data.get('organization', filename.replace('.json', ''))
                datasets[f"🌿 {org_name}"] = {
                    "path": filepath,
                    "type": "ecosystem",
                    "metadata": data.get('metadata', {})
                }
            except Exception as e:
                continue
    
    # Load user-saved datasets
    user_dir = "data/user_saved_networks"
    if os.path.exists(user_dir):
        user_files = [f for f in os.listdir(user_dir) if f.endswith('.json')]
        user_files.sort(reverse=True)  # Most recent first
        
        for filename in user_files:
            try:
                filepath = os.path.join(user_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract display name and metadata
                org_name = data.get('organization', filename.replace('.json', ''))
                metadata = data.get('metadata', {})
                created = metadata.get('created', 'Unknown')
                network_type = metadata.get('network_description', 'Generated Network')
                
                # Format display name
                display_name = f"💾 {org_name} ({network_type})"
                if created != 'Unknown':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d %H:%M')
                        display_name += f" - {date_str}"
                    except:
                        pass
                
                datasets[display_name] = {
                    "path": filepath, 
                    "type": "user_saved",
                    "metadata": metadata
                }
                
            except Exception as e:
                continue  # Skip corrupted files
    
    return datasets

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

def show_analysis_page():
    """Show the analysis page with network visualization on top and sidebar navigation."""
    
    # Get analysis data from session state
    if st.session_state.analysis_data is None:
        st.error("No analysis data available")
        if st.button("↩️ Back to Main"):
            st.session_state.current_page = 'main'
            st.rerun()
        return
    
    data = st.session_state.analysis_data
    flow_matrix = data['flow_matrix']
    node_names = data['node_names']
    org_name = data['org_name']
    
    # Calculate metrics
    with st.spinner("Calculating sustainability metrics..."):
        calculator = UlanowiczCalculator(flow_matrix, node_names)
        extended_metrics = calculator.get_extended_metrics()
        assessments = calculator.assess_regenerative_health()
    
    # Create the network graph for visualizations (will be used in Network Analysis section)
    if 'network' in data and data['network'] is not None:
        # Use existing network graph if available (for synthetic data)
        G = data['network']
    else:
        # Create network graph from flow matrix (for sample/uploaded data)
        import networkx as nx
        G = nx.DiGraph()
        for i, node in enumerate(node_names):
            G.add_node(i, name=node)
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if flow_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=flow_matrix[i, j])
    
    # Store network in session state for access in Network Analysis
    st.session_state.analysis_data['network'] = G
    
    # Sidebar with back button at the top
    if st.sidebar.button("← Back to Data Selection", type="primary", use_container_width=True):
        st.session_state.current_page = 'main'
        st.session_state.analysis_data = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Sidebar navigation for detailed analysis
    st.sidebar.title("📊 Analysis Sections")
    analysis_section = st.sidebar.radio(
        "Choose Analysis View:",
        ["🎯 Core Metrics", "🔄 Network Analysis", "📊 Visualizations", "📋 Detailed Report"]
    )
    
    # Display selected section
    if analysis_section == "🎯 Core Metrics":
        display_core_metrics_combined(extended_metrics, assessments, org_name, flow_matrix, node_names)
    elif analysis_section == "🔄 Network Analysis":
        display_network_analysis(calculator, extended_metrics, flow_matrix, node_names)
    elif analysis_section == "📊 Visualizations":
        display_visualizations_enhanced(G, flow_matrix, node_names, extended_metrics)
    elif analysis_section == "📋 Detailed Report":
        display_detailed_report(calculator, extended_metrics, assessments, org_name)

def run_analysis(flow_matrix, node_names, org_name):
    """Legacy function - redirect to new analysis page."""
    st.session_state.analysis_data = {
        'flow_matrix': flow_matrix,
        'node_names': node_names,
        'org_name': org_name,
        'source': 'direct'
    }
    st.session_state.current_page = 'analysis'
    st.rerun()

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

def display_visualizations_enhanced(G, flow_matrix, node_names, metrics):
    """Display visualizations with network diagram, flow heatmap, and window of viability."""
    
    st.header("📊 Visualizations")
    
    # Network Visualization first
    st.subheader("🌐 Network Diagram")
    
    # Display network visualization
    if len(node_names) <= 200:
        from network_generator import OrganizationalNetworkGenerator
        generator = OrganizationalNetworkGenerator()
        fig = generator.create_plotly_visualization(
            G, node_names,
            title=f"Network Structure - {len(node_names)} nodes, {G.number_of_edges()} edges"
        )
        fig.update_layout(height=700)  # Large visualization
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"🏢 Network too large for visualization ({len(node_names)} nodes).")
        degrees = [G.degree(n) for n in G.nodes()]
        degree_fig = px.histogram(
            x=degrees, 
            title=f"Degree Distribution",
            labels={"x": "Node Degree", "y": "Count"}
        )
        st.plotly_chart(degree_fig, use_container_width=True)
    
    # Network Flow Heatmap second
    st.subheader("🔥 Network Flow Heatmap")
    flow_fig = create_flow_heatmap(flow_matrix, node_names)
    st.plotly_chart(flow_fig, use_container_width=True)
    
    # Window of Viability last
    st.subheader("🎯 Window of Viability")
    robustness_fig = create_robustness_curve(metrics)
    st.plotly_chart(robustness_fig, use_container_width=True)
    
    # Flow Statistics
    st.subheader("📊 Flow Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flow", f"{np.sum(flow_matrix):.1f}")
        st.metric("Active Connections", np.count_nonzero(flow_matrix))
    with col2:
        st.metric("Avg Flow", f"{np.mean(flow_matrix[flow_matrix > 0]):.2f}" if np.any(flow_matrix > 0) else "0")
        st.metric("Median Flow", f"{np.median(flow_matrix[flow_matrix > 0]):.2f}" if np.any(flow_matrix > 0) else "0")
    with col3:
        st.metric("Max Flow", f"{np.max(flow_matrix):.1f}")
        st.metric("Min Flow (>0)", f"{np.min(flow_matrix[flow_matrix > 0]):.2f}" if np.any(flow_matrix > 0) else "0")
    with col4:
        flow_std = np.std(flow_matrix[flow_matrix > 0]) if np.any(flow_matrix > 0) else 0
        st.metric("Flow Std Dev", f"{flow_std:.2f}")
        flow_cv = flow_std / np.mean(flow_matrix[flow_matrix > 0]) if np.any(flow_matrix > 0) and np.mean(flow_matrix[flow_matrix > 0]) > 0 else 0
        st.metric("Coeff. of Variation", f"{flow_cv:.2f}")

def display_core_metrics_combined(metrics, assessments, org_name, flow_matrix, node_names):
    """Display combined core metrics including Ulanowicz and Regenerative indicators."""
    
    # Core Metrics header at the top
    st.header("🎯 Core Metrics")
    
    # Network name and summary
    st.markdown(f"### 🌐 {org_name}")
    
    # Quick summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Nodes", len(node_names))
    with col2:
        st.metric("Total Flow", f"{np.sum(flow_matrix):.1f}")
    with col3:
        st.metric("Relative Ascendency", f"{metrics['relative_ascendency']:.3f}")
    with col4:
        st.metric("Robustness", f"{metrics['robustness']:.3f}")
    with col5:
        viable = "✅ YES" if metrics['is_viable'] else "❌ NO"
        st.metric("Viable", viable)
    
    st.markdown("---")
    
    # Section 1: Overview
    st.subheader("📊 Overview")
    st.markdown("### System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Relative Ascendency", f"{metrics['relative_ascendency']:.3f}")
        st.caption("Organization level (α)")
    
    with col2:
        st.metric("Robustness", f"{metrics['robustness']:.3f}")
        st.caption("Resilience to shocks")
    
    with col3:
        viable = "✅ YES" if metrics['is_viable'] else "❌ NO"
        st.metric("Viable System", viable)
        st.caption("Within sustainability bounds")
    
    with col4:
        st.metric("Network Efficiency", f"{metrics['network_efficiency']:.3f}")
        st.caption("Resource utilization")
    
    # Sustainability assessment
    st.markdown("---")
    st.subheader("🌱 Sustainability Assessment")
    
    ascendency = metrics['ascendency']
    lower = metrics['viability_lower_bound']
    upper = metrics['viability_upper_bound']
    
    if lower <= ascendency <= upper:
        if ascendency < (lower + upper) / 2:
            st.success("✅ VIABLE - System is sustainable with good flexibility")
        else:
            st.success("✅ VIABLE - System is sustainable with good organization")
    elif ascendency < lower:
        st.error("❌ UNSUSTAINABLE - System is too chaotic (low organization)")
        st.info("💡 Recommendation: Increase structure and coordination")
    else:
        st.error("❌ UNSUSTAINABLE - System is too rigid (over-organized)")
        st.info("💡 Recommendation: Increase flexibility and redundancy")
    
    
    # Section 2: Ulanowicz Indicators
    st.markdown("---")
    st.subheader("📈 Ulanowicz Indicators")
    
    st.markdown("""
    Fundamental metrics from Robert Ulanowicz's Information Theory approach to ecosystem analysis,
    adapted for organizational networks.
    """)
    
    # Main indicators
    st.markdown("### 🔄 System Activity Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total System Throughput (TST)", f"{metrics['total_system_throughput']:.1f}")
        st.caption("Total flow/activity in the network")
        
        st.metric("Average Mutual Information (AMI)", f"{metrics['average_mutual_information']:.3f}")
        st.caption("Degree of organization in flow patterns")
        
        st.metric("Ascendency (A)", f"{metrics['ascendency']:.1f}")
        st.caption("Organized power (TST × AMI)")
    
    with col2:
        st.metric("Development Capacity (C)", f"{metrics['development_capacity']:.1f}")
        st.caption("Maximum possible organization")
        
        st.metric("Overhead/Reserve (Φ)", f"{metrics['overhead']:.1f}")
        st.caption("Unutilized capacity (C - A)")
        
        st.metric("Flow Diversity (H)", f"{metrics['flow_diversity']:.3f}")
        st.caption("Shannon entropy of flows")
    
    # Fundamental relationship
    st.markdown("---")
    st.markdown("### ⚖️ Fundamental Relationship")
    
    # Verify C = A + Φ
    c = metrics['development_capacity']
    a = metrics['ascendency']
    phi = metrics['overhead']
    calculated = a + phi
    error = abs(c - calculated)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("C (Capacity)", f"{c:.1f}")
    with col2:
        st.metric("A + Φ", f"{calculated:.1f}")
    with col3:
        if error < 0.01:
            st.success(f"✅ Error: {error:.4f}")
        else:
            st.warning(f"⚠️ Error: {error:.4f}")
    
    st.caption("Fundamental IT relationship: C = A + Φ (Capacity = Ascendency + Overhead)")
    
    
    # Section 3: Regenerative Metrics
    st.markdown("---")
    st.subheader("🌱 Regenerative Metrics")
    
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
    st.markdown("---")
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
    
    
    # Section 4: Balance Analysis
    st.markdown("---")
    st.subheader("⚖️ Balance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Ascendency to Capacity ratio gauge
        ratio = metrics['ascendency_ratio']
        if ratio < 0.2:
            color = "red"
            status = "Too Chaotic"
        elif ratio > 0.6:
            color = "red"
            status = "Too Rigid"
        elif 0.35 <= ratio <= 0.4:
            color = "green"
            status = "Optimal"
        else:
            color = "#f59e0b"
            status = "Acceptable"
        
        st.metric("Organization Ratio (A/C)", f"{ratio:.3f}")
        st.markdown(f"Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        st.progress(ratio)
        st.caption("Degree of organization")
    
    with col2:
        # Overhead ratio
        overhead_ratio = metrics['overhead_ratio']
        st.metric("Flexibility Ratio (Φ/C)", f"{overhead_ratio:.3f}")
        if overhead_ratio < 0.4:
            st.markdown("Status: <span style='color:red'>Low Reserve</span>", unsafe_allow_html=True)
        elif overhead_ratio > 0.8:
            st.markdown("Status: <span style='color:#f59e0b'>High Redundancy</span>", unsafe_allow_html=True)
        else:
            st.markdown("Status: <span style='color:green'>Good Balance</span>", unsafe_allow_html=True)
        st.progress(overhead_ratio)
        st.caption("Reserve capacity")
    
    # Window of Viability details
    st.markdown("---")
    st.subheader("🎯 Window of Viability Analysis")
    
    lower = metrics['viability_lower_bound']
    upper = metrics['viability_upper_bound']
    current = metrics['ascendency']
    
    # Visual representation
    progress_val = (current - lower) / (upper - lower) if upper > lower else 0.5
    progress_val = max(0, min(1, progress_val))  # Clamp between 0 and 1
    
    st.progress(progress_val)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound", f"{lower:.1f}")
        st.caption("20% of capacity")
    with col2:
        st.metric("Current Position", f"{current:.1f}")
        if lower <= current <= upper:
            st.caption("✅ Within bounds")
        else:
            st.caption("❌ Outside bounds")
    with col3:
        st.metric("Upper Bound", f"{upper:.1f}")
        st.caption("60% of capacity")
    
    # Efficiency vs Redundancy balance
    st.markdown("---")
    balance = metrics['ascendency_ratio'] / (metrics['overhead_ratio'] + 0.001)
    st.metric("Efficiency/Redundancy Balance", f"{balance:.2f}")
    if 0.5 <= balance <= 2:
        st.info("✅ System has good balance between efficiency and redundancy")
    elif balance < 0.5:
        st.warning("⚠️ System is too redundant - consider streamlining processes")
    else:
        st.warning("⚠️ System is too efficient - consider adding backup pathways")

def display_core_metrics_simplified(metrics):
    """Display simplified core metrics."""
    
    st.header("🎯 Core Metrics")
    
    # Quick overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Relative Ascendency", f"{metrics['relative_ascendency']:.3f}")
        st.caption("Organization level (α)")
    
    with col2:
        st.metric("Robustness", f"{metrics['robustness']:.3f}")
        st.caption("Resilience to shocks")
    
    with col3:
        viable = "✅ YES" if metrics['is_viable'] else "❌ NO"
        st.metric("Viable System", viable)
        st.caption("Within sustainability bounds")
    
    with col4:
        st.metric("Network Efficiency", f"{metrics['network_efficiency']:.3f}")
        st.caption("Resource utilization")
    
    # Sustainability assessment
    st.markdown("---")
    st.subheader("🌱 Sustainability Assessment")
    
    ascendency = metrics['ascendency']
    lower = metrics['viability_lower_bound']
    upper = metrics['viability_upper_bound']
    
    if lower <= ascendency <= upper:
        if ascendency < (lower + upper) / 2:
            st.success("✅ VIABLE - System is sustainable with good flexibility")
        else:
            st.success("✅ VIABLE - System is sustainable with good organization")
    elif ascendency < lower:
        st.error("❌ UNSUSTAINABLE - System is too chaotic (low organization)")
        st.info("💡 Recommendation: Increase structure and coordination")
    else:
        st.error("❌ UNSUSTAINABLE - System is too rigid (over-organized)")
        st.info("💡 Recommendation: Increase flexibility and redundancy")
    
    # Key ratios
    st.markdown("---")
    st.subheader("📊 Balance Indicators")
    
    col1, col2 = st.columns(2)
    with col1:
        # Ascendency to Capacity ratio gauge
        ratio = metrics['ascendency_ratio']
        if ratio < 0.2:
            color = "red"
            status = "Too Chaotic"
        elif ratio > 0.6:
            color = "red"
            status = "Too Rigid"
        elif 0.35 <= ratio <= 0.4:
            color = "green"
            status = "Optimal"
        else:
            color = "#f59e0b"
            status = "Acceptable"
        
        st.metric("Organization Ratio (A/C)", f"{ratio:.3f}")
        st.markdown(f"Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    
    with col2:
        # Overhead ratio
        overhead_ratio = metrics['overhead_ratio']
        st.metric("Flexibility Ratio (Φ/C)", f"{overhead_ratio:.3f}")
        if overhead_ratio < 0.4:
            st.markdown("Status: <span style='color:red'>Low Reserve</span>", unsafe_allow_html=True)
        elif overhead_ratio > 0.8:
            st.markdown("Status: <span style='color:#f59e0b'>High Redundancy</span>", unsafe_allow_html=True)
        else:
            st.markdown("Status: <span style='color:green'>Good Balance</span>", unsafe_allow_html=True)

def display_ulanowicz_indicators(metrics):
    """Display detailed Ulanowicz indicators."""
    
    st.header("📈 Core Ulanowicz Indicators")
    
    st.markdown("""
    These are the fundamental metrics from Robert Ulanowicz's Information Theory approach to ecosystem analysis,
    adapted for organizational networks.
    """)
    
    # Main indicators
    st.subheader("🔄 System Activity Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total System Throughput (TST)", f"{metrics['total_system_throughput']:.1f}")
        st.caption("Total flow/activity in the network")
        
        st.metric("Average Mutual Information (AMI)", f"{metrics['average_mutual_information']:.3f}")
        st.caption("Degree of organization in flow patterns")
        
        st.metric("Ascendency (A)", f"{metrics['ascendency']:.1f}")
        st.caption("Organized power (TST × AMI)")
    
    with col2:
        st.metric("Development Capacity (C)", f"{metrics['development_capacity']:.1f}")
        st.caption("Maximum possible organization")
        
        st.metric("Overhead/Reserve (Φ)", f"{metrics['overhead']:.1f}")
        st.caption("Unutilized capacity (C - A)")
        
        st.metric("Flow Diversity (H)", f"{metrics['flow_diversity']:.3f}")
        st.caption("Shannon entropy of flows")
    
    # Fundamental relationship
    st.markdown("---")
    st.subheader("⚖️ Fundamental Relationship")
    
    # Verify C = A + Φ
    c = metrics['development_capacity']
    a = metrics['ascendency']
    phi = metrics['overhead']
    calculated = a + phi
    error = abs(c - calculated)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("C (Capacity)", f"{c:.1f}")
    with col2:
        st.metric("A + Φ", f"{calculated:.1f}")
    with col3:
        if error < 0.01:
            st.success(f"✅ Error: {error:.4f}")
        else:
            st.warning(f"⚠️ Error: {error:.4f}")
    
    st.caption("Fundamental IT relationship: C = A + Φ (Capacity = Ascendency + Overhead)")
    
    # Ratios and percentages
    st.markdown("---")
    st.subheader("📊 Key Ratios")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ascendency Ratio (α = A/C)", f"{metrics['ascendency_ratio']:.3f}")
        st.progress(metrics['ascendency_ratio'])
        st.caption("Degree of organization")
    
    with col2:
        st.metric("Overhead Ratio (Φ/C)", f"{metrics['overhead_ratio']:.3f}")
        st.progress(metrics['overhead_ratio'])
        st.caption("Reserve capacity")
    
    with col3:
        # Efficiency vs Redundancy balance
        balance = metrics['ascendency_ratio'] / (metrics['overhead_ratio'] + 0.001)
        st.metric("Efficiency/Redundancy", f"{balance:.2f}")
        if 0.5 <= balance <= 2:
            st.caption("✅ Good balance")
        else:
            st.caption("⚠️ Imbalanced")
    
    # Window of Viability details
    st.markdown("---")
    st.subheader("🎯 Window of Viability Analysis")
    
    lower = metrics['viability_lower_bound']
    upper = metrics['viability_upper_bound']
    current = metrics['ascendency']
    
    # Visual representation
    progress_val = (current - lower) / (upper - lower) if upper > lower else 0.5
    progress_val = max(0, min(1, progress_val))  # Clamp between 0 and 1
    
    st.progress(progress_val)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound", f"{lower:.1f}")
        st.caption("20% of capacity")
    with col2:
        st.metric("Current Position", f"{current:.1f}")
        if lower <= current <= upper:
            st.caption("✅ Within bounds")
        else:
            st.caption("❌ Outside bounds")
    with col3:
        st.metric("Upper Bound", f"{upper:.1f}")
        st.caption("60% of capacity")

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
    """Display network analysis metrics and properties."""
    
    st.header("🔄 Network Analysis")
    
    # Get or create the network graph for metrics calculation
    if st.session_state.analysis_data and 'network' in st.session_state.analysis_data and st.session_state.analysis_data['network'] is not None:
        G = st.session_state.analysis_data['network']
    else:
        # Create network graph from flow matrix
        import networkx as nx
        G = nx.DiGraph()
        for i, node in enumerate(node_names):
            G.add_node(i, name=node)
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if flow_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=flow_matrix[i, j])
    
    # Network Properties
    st.subheader("🔗 Network Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Structure Metrics")
        st.write(f"**Nodes**: {calculator.n_nodes}")
        st.write(f"**Edges**: {G.number_of_edges()}")
        st.write(f"**Total Connections**: {np.count_nonzero(flow_matrix)}")
        st.write(f"**Network Density**: {np.count_nonzero(flow_matrix) / (calculator.n_nodes ** 2):.3f}")
        st.write(f"**Avg Degree**: {2 * G.number_of_edges() / calculator.n_nodes:.2f}")
    
    with col2:
        st.markdown("### Flow Metrics")
        st.write(f"**Total Flow**: {np.sum(flow_matrix):.1f}")
        st.write(f"**Average Flow**: {np.mean(flow_matrix[flow_matrix > 0]):.1f}")
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
    """Educational interface about adaptive organizations and regenerative economics."""
    
    st.header("📚 Learn About Adaptive Organizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌱 Adaptive Organizations", "📊 Key Metrics", "🎯 Window of Viability", 
        "🔬 Scientific Foundation", "🚀 Practical Applications"
    ])
    
    with tab1:
        st.markdown("""
        ## 🌱 Adaptive Organizations Framework
        
        This analysis system combines **Massimo Mistretta's Adaptive Organizations** methodology with 
        **Robert Ulanowicz's Ecosystem Theory** and **Regenerative Economics** principles.
        
        ### Core Philosophy: Organizations as Living Systems
        
        Adaptive organizations are **living systems** that continuously evolve, learn, and regenerate. 
        Like ecosystems, they must balance **efficiency** with **resilience** to thrive in complex environments.
        
        ### Key Principles of Adaptive Organizations
        
        **🔄 Continuous Adaptation**
        - Organizations must constantly evolve to survive
        - Change is not episodic but continuous
        - Adaptation happens at all levels simultaneously
        
        **🌐 Network-Centric Design**  
        - Organizations are networks of relationships and flows
        - Information, resources, and energy flow through the system
        - Network topology determines organizational capabilities
        
        **⚖️ Balance Over Optimization**
        - Avoid over-optimization in any single dimension
        - Maintain healthy tension between order and chaos
        - Preserve capacity for future adaptation
        
        **🎯 Purpose-Driven Evolution**
        - Clear purpose guides adaptive choices
        - Values and mission provide stability amid change
        - Evolution serves larger organizational purpose
        
        **🔬 Evidence-Based Adaptation**
        - Use data and metrics to guide adaptation
        - Scientific approach to organizational design
        - Measure what matters for long-term sustainability
        
        ### The Adaptive Advantage
        
        Organizations that master adaptive principles:
        - **Thrive in uncertainty** rather than just survive it
        - **Regenerate capabilities** faster than they consume them  
        - **Create value** for all stakeholders, not just shareholders
        - **Evolve with their environment** rather than resist change
        
        ---
        
        *Learn more: [Adaptive Organizations on Medium](https://medium.com/adaptive-organizations)*
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
        ## 🔬 Scientific Foundation
        
        ### Primary Research Sources
        
        **🌿 Robert E. Ulanowicz - Ecosystem Theory**
        - "A Third Window: Natural Life Beyond Newton and Darwin"  
        - "Quantifying Sustainability: Resilience, Efficiency and the Return of Information Theory"
        - Pioneer of network-based sustainability analysis
        - Developer of ascendency and window of viability concepts
        
        **🔄 Brian D. Fath & Ulanowicz - Regenerative Economics**
        - "Measuring Regenerative Economics: 10 principles and measures undergirding systemic economic health"
        - Extended ecosystem theory to economic systems
        - Developed regenerative capacity framework
        
        **🏢 Massimo Mistretta - Adaptive Organizations**
        - [Adaptive Organizations Medium Publication](https://medium.com/adaptive-organizations)
        - Integration of ecosystem theory with organizational design
        - Practical application of regenerative principles to business
        - Evidence-based approach to organizational transformation
        
        ### Mathematical Foundations
        
        **Information Theory**
        - Shannon entropy for measuring system organization
        - Mutual information for quantifying relationships
        - Mathematical precision in sustainability measurement
        
        **Network Science**
        - Graph theory for organizational structure analysis
        - Flow dynamics and system throughput
        - Centrality and connectivity measures
        
        **Complex Systems Theory**
        - Self-organization and emergence
        - Non-linear dynamics and phase transitions
        - Adaptive capacity and resilience
        """)
    
    with tab5:
        st.markdown("""
        ## 🚀 Practical Applications
        
        ### Organizational Design & Transformation
        
        **🏗️ Structure Design**
        - Use network visualization to design optimal structures
        - Balance centralization and distribution based on scientific metrics
        - Create adaptive communication patterns that enhance sustainability
        
        **🔄 Change Management**
        - Monitor sustainability metrics during organizational transitions
        - Maintain position in window of viability throughout change
        - Preserve adaptive capacity while implementing improvements
        
        **📊 Performance Management**
        - Track regenerative capacity alongside traditional KPIs
        - Use robustness as leading indicator of organizational health
        - Measure network effects of management decisions
        
        ### Strategic Leadership Applications
        
        **🎯 Strategic Planning**
        - Assess organizational readiness for strategic initiatives
        - Design strategies that enhance rather than consume adaptive capacity
        - Balance growth ambitions with long-term sustainability
        
        **🤝 Merger & Integration**
        - Analyze cultural and structural compatibility before mergers
        - Design integration processes that preserve key capabilities
        - Monitor network health throughout integration process
        
        **💡 Innovation Ecosystem Design**
        - Create conditions for sustainable innovation
        - Balance exploration (new ideas) with exploitation (current strengths)
        - Design innovation networks for maximum adaptability
        
        ### Using This Analysis Tool
        
        **🔍 Diagnose** your organization's current sustainability position
        **🎨 Design** new organizational structures and test their viability scientifically  
        **📊 Monitor** organizational health metrics over time
        **🧪 Experiment** with different network configurations safely
        **📚 Learn** from successful adaptive organization patterns in nature
        **🚀 Transform** your organization into a regenerative adaptive system
        
        ### Implementation Roadmap
        
        1. **Current State Assessment**
           - Map existing communication and information flows
           - Calculate baseline sustainability metrics
           - Identify position in window of viability
        
        2. **Vision Design** 
           - Use network generator to explore optimal structures
           - Test different scenarios and their sustainability implications
           - Save promising configurations to your library
        
        3. **Gradual Transformation**
           - Implement changes that move toward adaptive principles
           - Monitor metrics during transformation process
           - Preserve what works while evolving what doesn't
        
        4. **Continuous Evolution**
           - Regular sustainability assessments
           - Adaptive adjustments based on changing conditions
           - Culture of evidence-based continuous improvement
        
        ---
        
        *Ready to transform your organization? Start with the Visual Network Generator to explore possibilities.*
        """)

def ten_principles_interface():
    """Display the 10 Principles of Regenerative Economics."""
    
    st.header("🌱 10 Principles of Regenerative Economics")
    st.markdown("""
    These principles, developed by **Fath, Fiscus, Goerner, Berea & Ulanowicz (2019)**, provide a comprehensive 
    framework for understanding and measuring systemic economic health based on decades of research in 
    ecological network analysis and complex systems science.
    """)
    
    # Add reference with link
    st.info("""
    📚 **Source**: Fath, B.D., Fiscus, D.A., Goerner, S.J., Berea, A., & Ulanowicz, R.E. (2019). 
    "Measuring regenerative economics: 10 principles and measures undergirding systemic economic health." 
    *Global Transitions*, 1, 15-27.
    """)
    
    # Create four main categories as expandable sections
    with st.expander("🔄 **CIRCULATION** (Principles 1-4)", expanded=True):
        st.markdown("""
        ### **Principle 1: Maintain Robust Cross-Scale Circulation**
        - **What**: Ensure money, information, resources flow across all scales
        - **Why**: All sectors and levels play mutually supportive, interlinked roles
        - **Measure**: Network Aggradation = TST/Σzi (Total System Throughput / Total Inputs)
        - **Example**: Low wages reduce circulation → economic necrosis
        
        ### **Principle 2: Regenerative Re-Investment**
        - **What**: Continuously invest in human, social, natural, and physical capital
        - **Why**: Systems must be self-nourishing and self-renewing to thrive
        - **Measure**: Finn Cycling Index (FCI) = ΣTci/TST
        - **Example**: Every $1 on GI Bill returned $7 to economy
        
        ### **Principle 3: Maintain Reliable Inputs**
        - **What**: Ensure steady supply of critical resources (energy, water, information)
        - **Why**: Systems collapse without essential inputs
        - **Measure**: % renewable energy, EROI trends, supply chain resilience
        - **Example**: Fossil fuel dependency creates systemic vulnerability
        
        ### **Principle 4: Maintain Healthy Outputs**
        - **What**: Minimize harmful waste and environmental damage
        - **Why**: Systems that foul their environment cannot survive
        - **Measure**: Pollution levels, carbon sequestration capacity
        - **Example**: Circular economy principles, zero-waste initiatives
        """)
    
    with st.expander("🏗️ **STRUCTURE** (Principles 5-6)", expanded=True):
        st.markdown("""
        ### **Principle 5: Balance Small, Medium & Large Organizations**
        - **What**: Maintain fractal/power-law distribution of organizational sizes
        - **Why**: Each scale serves unique functions; imbalance creates brittleness
        - **Measure**: Compare size distribution against xⁿ power-law patterns
        - **Example**: Too many "too-big-to-fail" banks → 2008 crisis
        
        ### **Principle 6: Balance Resilience and Efficiency**
        - **What**: Maintain optimal trade-off between streamlining and redundancy
        - **Why**: Too much efficiency → brittleness; too much redundancy → stagnation
        - **Measure**: Robustness = -α·log(α) where α = A/C
        - **Window of Vitality**: 0.2 < α < 0.6 (optimal sustainability range)
        """)
    
    with st.expander("🤝 **RELATIONSHIPS & VALUES** (Principles 7-8)", expanded=True):
        st.markdown("""
        ### **Principle 7: Maintain Sufficient Diversity**
        - **What**: Ensure adequate variety of roles, functions, and specialists
        - **Why**: Diversity enables filling niches and finding new solutions
        - **Measure**: Number of functional roles = Π(Fij·F../Fi·F·j)^(Fij/F..)
        - **Example**: Monocultures are vulnerable; diverse ecosystems are resilient
        
        ### **Principle 8: Promote Mutually-Beneficial Relationships**
        - **What**: Foster cooperation and common-cause values over pure competition
        - **Why**: Collaboration produces more than isolated self-interest
        - **Measure**: Ratio of mutualistic (+,+) to exploitative (+,-) relationships
        - **Example**: Trust, justice, and reciprocity enhance economic vitality
        """)
    
    with st.expander("📚 **COLLECTIVE LEARNING** (Principles 9-10)", expanded=True):
        st.markdown("""
        ### **Principle 9: Promote Constructive over Extractive Activity**
        - **What**: Build value and capacity rather than extract existing wealth
        - **Why**: Extraction without regeneration leads to systemic decline
        - **Measure**: Ratio of value-add activities to speculation/extraction
        - **Example**: Real economy investment vs. financial speculation
        
        ### **Principle 10: Promote Effective Collective Learning**
        - **What**: Enable society-wide adaptation and knowledge evolution
        - **Why**: Learning is humanity's core survival strategy
        - **Measure**: Education investment, innovation indices, civic engagement
        - **Example**: Societies that stop learning eventually collapse
        """)
    
    # Add practical application section
    st.markdown("---")
    st.subheader("🎯 Practical Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **For Organizations**
        - Assess circulation patterns
        - Balance efficiency with resilience
        - Invest in human capital
        - Foster collaborative culture
        - Measure constructive vs extractive activities
        """)
    
    with col2:
        st.markdown("""
        ### **For Policymakers**
        - Support cross-scale circulation
        - Prevent excessive concentration
        - Incentivize regenerative investment
        - Maintain diversity of enterprises
        - Enable collective learning systems
        """)
    
    # Add key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    st.success("""
    **The Window of Vitality**: Natural systems teach us that sustainability requires balancing 
    efficiency (α ≈ 0.37) within a viable range (0.2 < α < 0.6). Too much order leads to brittleness; 
    too much chaos leads to stagnation. The sweet spot enables both productivity and adaptability.
    """)
    
    st.warning("""
    **Current Challenge**: Most modern economies over-emphasize efficiency and size (Principles 5 & 6), 
    leading to systemic fragility. The 2008 financial crisis exemplified this imbalance. 
    Regenerative economics seeks to restore healthy balance across all 10 principles.
    """)
    
    st.info("""
    **Path Forward**: These principles are not just idealistic goals but measurable, 
    scientifically-grounded metrics. By tracking and optimizing these indicators, 
    organizations and societies can build truly sustainable, regenerative systems.
    """)

def formulas_reference_interface():
    """Complete formulas reference for all indicators."""
    
    st.header("🔬 Complete Formulas Reference")
    st.markdown("""
    This page contains all mathematical formulations used in the Adaptive Organization Analysis system,
    organized by category and based on peer-reviewed scientific literature.
    """)
    
    # Create tabs for different categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧮 Core Ulanowicz IT", "💚 Regenerative Health", "🌱 Regenerative Economics", 
        "📊 Network Analysis", "🎯 Sustainability Metrics", "⚙️ Extended Indicators"
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
        st.subheader("💚 Regenerative Health Metrics")
        st.markdown("*Key formulas for measuring systemic health based on the 10 principles of regenerative economics*")
        
        st.info("📚 **Source**: Fath et al. (2019) - 'Measuring regenerative economics: 10 principles and measures undergirding systemic economic health', Global Transitions, 1, 15-27.")
        
        st.markdown("""
        ### **1. Network Aggradation (Cross-Scale Circulation)**
        ```
        Network Aggradation = TST / Σzi
        ```
        - **Principle 1**: Robust cross-scale circulation
        - Ratio of Total System Throughput to total inputs
        - Higher values indicate more internal circulation
        
        ### **2. Finn Cycling Index (Regenerative Re-investment)**
        ```
        FCI = ΣTci / TST
        where Tci = ((nii - 1) / nii) × Ti
        ```
        - **Principle 2**: Regenerative re-investment
        - Fraction of total flow that is recycled
        - nii = path multiplier from i back to i
        
        ### **3. Ascendency (A) - Organization Measure**
        ```
        A = Σ(Fij × log(Fij × F.. / (Fi. × F.j)))
        ```
        - Core measure of system organization
        - Fij = flow from i to j
        - F.. = total system flow
        
        ### **4. Development Capacity (C) - System Potential**
        ```
        C = -Σ(Fij × log(Fij / F..))
        ```
        - Maximum potential for development
        - Upper bound on system organization
        
        ### **5. Robustness (R) - System Health**
        ```
        Robustness = -α × log(α)
        where α = A/C
        ```
        - **Principle 6**: Balance of efficiency & resilience
        - Peak robustness at α ≈ 0.37
        - Window of Vitality: 0.2 < α < 0.6
        
        ### **6. Functional Diversity (Roles)**
        ```
        Roles = Π((Fij × F.. / (Fi. × F.j))^(Fij/F..))
        ```
        - **Principle 7**: Sufficient diversity
        - Number of distinct functional roles
        - Product over all non-zero flows
        
        ### **7. Mutualism Index**
        ```
        Direct Effects Matrix: [Dij]
        Total Effects (direct + indirect): N = Σ(B^m)
        Mutualism = Count(Nij > 0 AND Nji > 0) / Total pairs
        ```
        - **Principle 8**: Mutually-beneficial relationships
        - Ratio of mutualistic to total relationships
        
        ### **8. Constructive/Extractive Ratio**
        ```
        C/E Ratio = Value-Add Activities / Extractive Activities
        ```
        - **Principle 9**: Constructive over extractive
        - Distinguishes building from extracting
        
        ### **9. Average Mutual Information (AMI)**
        ```
        AMI = Σ(Fij × log(Fij × F.. / (Fi. × F.j))) / F..
        ```
        - Degree of constraint in the network
        - Normalized measure of organization
        
        ### **10. Window of Vitality Assessment**
        ```
        if α < 0.2: "Too little diversity" (Brittleness)
        if 0.2 ≤ α ≤ 0.6: "Sustainable balance"
        if α > 0.6: "Too little efficiency" (Stagnation)
        ```
        - Empirically validated bounds
        - Based on ecosystem observations
        """)
    
    with tab3:
        st.subheader("🌱 Regenerative Economics Formulations")
        st.markdown("*Extended formulations for regenerative capacity assessment*")
        
        st.markdown("""
        ### **Regenerative Capacity**
        ```
        RC = Robustness × (1 - |α - α_opt|)
        where α_opt = 0.37
        ```
        - Combines robustness with distance from optimum
        - Measures self-renewal potential
        
        ### **Flow Diversity (Shannon Entropy)**
        ```
        H = -Σ(pij × log(pij))
        where pij = Tij / TST
        ```
        - Evenness of flow distribution
        - Higher values = more distributed flows
        
        ### **Structural Information**
        ```
        SI = log(n²) - H
        ```
        - Network constraint independent of magnitudes
        - n = number of nodes
        
        ### **Redundancy Measure**
        ```
        Redundancy = Φ / C = 1 - α
        ```
        - Alternative pathways and backup capacity
        - Complement of efficiency
        
        ### **Effective Link Density**
        ```
        ELD = (L_active / L_max) × (AMI / AMI_max)
        ```
        - Weighted connectivity measure
        - Accounts for both structure and flow
        """)
    
    with tab4:
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
    
    with tab5:
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
    
    with tab6:
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
        Version 2.1.1 - Formula Validation & Accuracy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_app_version()