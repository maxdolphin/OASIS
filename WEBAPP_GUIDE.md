# 🌐 Streamlit Web Application Guide

## 🚀 **Quick Start**

The Adaptive Organization Analysis web app is now available! You can access it in two ways:

### **Method 1: Using the Startup Script (Recommended)**
```bash
# Make the script executable (if not already done)
chmod +x run_webapp.sh

# Launch the web app
./run_webapp.sh
```

### **Method 2: Direct Streamlit Command**
```bash
streamlit run app.py
```

## 🌐 **Accessing the Web App**

Once launched, the app will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.178.170:8501 (accessible from other devices on your network)

## ✨ **Web App Features**

### **📊 Upload Data Mode**
- Upload JSON or CSV files with your organizational flow data
- Automatic data validation and preview
- Support for custom department names and flow matrices
- Real-time analysis as soon as data is uploaded

### **🧪 Sample Data Mode**
- Choose from pre-built sample organizations:
  - TechFlow Innovations (Combined Email + Document flows)
  - TechFlow Innovations (Email only)
  - TechFlow Innovations (Documents only) 
  - Balanced Test Organization
- One-click analysis with instant results
- Perfect for learning and demonstration

### **⚡ Synthetic Data Generator**
- **Interactive UI** for creating custom organizations
- **Configurable parameters**:
  - Organization name and type
  - Department selection (choose from 12 options)
  - Communication intensity (Low/Medium/High)
  - Document formality levels
  - Random seed for reproducibility
- **Real-time generation** and immediate analysis

### **📱 Interactive Dashboard**

#### **Metrics Overview**
- **4 Key Performance Cards**: Network Efficiency, Robustness, Viability, Regenerative Capacity
- **Color-coded status indicators** (🟢 Good, 🟡 Moderate, 🔴 Needs Attention)
- **Overall system health assessment**

#### **5 Analysis Tabs**:

1. **🎯 Core Metrics Tab**
   - All traditional Ulanowicz indicators
   - Window of Viability visualization
   - Key ratios and system capacity metrics

2. **🌱 Regenerative Indicators Tab**
   - Extended regenerative economics metrics
   - Flow diversity and structural analysis
   - Health assessment breakdown with color coding

3. **📊 Visualizations Tab**
   - **🎯 Complete Sustainability Curve**: Shows theoretical curve with window of viability
     - Your organization's position prominently displayed
     - Reference organizations for comparison (Start-up, Mature Corp, etc.)
     - Clear zones: Chaos, Viable, and Rigidity zones
     - Optimal efficiency point (37%) marked
   - **💪 Enhanced Robustness Curve**: Shows efficiency vs. robustness relationship
     - Window of viability highlighted (20-60% efficiency)
     - Your position vs. theoretical optimum
   - **🔥 Network Flow Heatmap**: Department-to-department flow visualization
   - All charts fully interactive with zoom, pan, and detailed hover tooltips

4. **🔄 Network Analysis Tab**
   - Detailed network structure properties
   - Department-by-department flow analysis
   - Top 10 strongest connections ranking
   - Flow statistics and balance analysis

5. **📋 Detailed Report Tab**
   - **Comprehensive text report** with all metrics
   - **Executive summary** and recommendations
   - **Download options**:
     - 📄 Full report as TXT file
     - 📊 Analysis data as JSON file

### **📚 Learn More Section**
- **4 Educational Tabs**:
  - 🌱 **Overview**: Regenerative economics principles
  - 📊 **Key Metrics**: Detailed metric explanations
  - 🎯 **Window of Viability**: Theory and boundaries
  - 🔬 **Research**: Academic foundations and sources

## 🎨 **User Interface Features**

### **Responsive Design**
- **Wide layout** optimized for analysis dashboards
- **Collapsible sidebar** for easy navigation
- **Professional color scheme** with green sustainability theme

### **Interactive Elements**
- **File upload** with drag-and-drop support
- **Multi-select departments** for synthetic data
- **Slider controls** for intensity settings
- **Expandable sections** and tabs for organized content

### **Real-time Feedback**
- **Progress spinners** during analysis
- **Success/error messages** with clear guidance
- **Data preview** before analysis
- **Metric cards** with instant status updates

## 💾 **Export Capabilities**

### **Report Downloads**
- **Text Reports**: Complete analysis in readable format
- **JSON Data**: All metrics and raw data for further analysis
- **Custom naming**: Files named after your organization

### **Visualization Export**
- All Plotly charts are **interactive** in the browser
- **Hover tooltips** with detailed information
- **Zoom and pan** capabilities
- **Built-in screenshot** tools in each chart

## 🚀 **Performance Features**

### **Caching**
- **Streamlit caching** for faster repeated analysis
- **Efficient data processing** with numpy/pandas
- **Optimized visualizations** with Plotly

### **Error Handling**
- **Graceful error messages** for invalid data
- **Input validation** with helpful hints
- **Recovery suggestions** when things go wrong

## 📱 **Browser Compatibility**

**Fully Compatible**:
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile browsers (responsive design)
- Tablets and desktop computers

## 🛠️ **Technical Requirements**

### **Server Requirements**
- Python 3.8+
- 8GB RAM recommended for large organizations
- Modern web browser

### **Dependencies (Auto-installed)**
- streamlit
- plotly
- pandas
- numpy
- matplotlib
- networkx
- scipy

## 🔧 **Troubleshooting**

### **App Won't Start**
```bash
# Check if Streamlit is installed
pip3 show streamlit

# If not installed
pip3 install streamlit

# Try manual launch
cd /path/to/Adaptive_Organization
streamlit run app.py
```

### **Import Errors**
```bash
# Install all dependencies
pip3 install streamlit plotly pandas numpy matplotlib networkx scipy
```

### **Port Already in Use**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### **Browser Not Opening**
- Manually navigate to: http://localhost:8501
- Check firewall settings
- Try different browser

## 🎯 **Usage Tips**

### **For First-Time Users**
1. Start with **Sample Data** mode to explore features
2. Review the **Learn More** section for background
3. Try **Synthetic Data Generator** to understand parameters
4. Upload your own data when ready

### **For Data Upload**
- Ensure CSV files have proper headers
- JSON files should follow the example format
- Start with smaller datasets (< 20 departments) for testing
- Check data preview before running analysis

### **For Best Results**
- Use **detailed department names** for clarity
- Include **realistic flow values** (avoid zeros everywhere)
- Consider **flow meaning** (emails, documents, resources, etc.)
- Interpret results in **organizational context**

## 🆘 **Support**

If you encounter issues:
1. Check the **troubleshooting section** above
2. Verify your **data format** matches examples
3. Try with **sample data** first
4. Restart the application if needed

---

## 🎉 **Ready to Analyze!**

Your Streamlit web application is fully functional with:
- ✅ Interactive user interface
- ✅ File upload capabilities  
- ✅ Synthetic data generation
- ✅ Real-time analysis
- ✅ Comprehensive visualizations
- ✅ Export functionality
- ✅ Educational content

**Open your browser to http://localhost:8501 and start exploring!** 🌱