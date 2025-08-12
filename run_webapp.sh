#!/bin/bash

echo "🌱 Starting Adaptive Organization Analysis Web App..."
echo "=============================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Installing now..."
    pip3 install streamlit
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, numpy, pandas, plotly; print('✅ All dependencies are available')" || {
    echo "❌ Missing dependencies. Installing now..."
    pip3 install streamlit numpy pandas plotly matplotlib networkx scipy
}

echo "🚀 Launching web application..."
echo ""
echo "The app will be available at:"
echo "  🌐 Local URL: http://localhost:8501"
echo ""
echo "📱 Features available:"
echo "  • Upload your own organizational data"
echo "  • Analyze sample organizations"
echo "  • Generate synthetic data"
echo "  • Interactive visualizations"
echo "  • Comprehensive reports"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=============================================="

streamlit run app.py