# Contributing to Adaptive Organization Analysis

Thank you for your interest in contributing to the Adaptive Organization Analysis project! This project implements the Ulanowicz-Fath regenerative economics framework for organizational sustainability analysis.

## 🎯 Ways to Contribute

### 🐛 **Bug Reports**
- Check existing issues first
- Provide clear reproduction steps
- Include system information (Python version, OS, etc.)
- Attach sample data if relevant

### 💡 **Feature Requests**
- Describe the use case and benefits
- Consider how it fits with the theoretical framework
- Propose implementation approach if possible

### 🔬 **Research Contributions**
- New analysis methods based on network ecology
- Alternative visualization approaches
- Additional regenerative economics indicators
- Case studies and validation data

### 🎨 **Visualization Enhancements**
- Interactive chart improvements
- New chart types for organizational analysis
- UI/UX improvements for the Streamlit app
- Mobile-responsive design enhancements

### 📚 **Documentation**
- Code documentation and docstrings
- Tutorial improvements
- Use case examples
- Theoretical background explanations

## 🚀 **Getting Started**

### **Development Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/Adaptive_Organization.git
cd Adaptive_Organization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r docs/requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### **Running Tests**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ulanowicz_calculator.py

# Run with coverage
pytest --cov=src
```

### **Code Style**
- Use Black for code formatting: `black src/ tests/`
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for all public functions

## 📊 **Project Structure**

```
src/
├── ulanowicz_calculator.py  # Core calculations
├── visualizer.py           # Visualization system
└── main.py                 # CLI interface

data/synthetic_organizations/ # Sample datasets
docs/                        # Documentation
tests/                       # Test suite
```

## 🔬 **Theoretical Framework**

When contributing, please consider:

- **Ulanowicz's Ecosystem Theory**: Maintain theoretical consistency
- **Regenerative Economics**: Align with sustainability principles  
- **Network Ecology**: Respect information-theoretic foundations
- **Systems Thinking**: Consider emergent properties and feedback loops

## 📋 **Pull Request Process**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Test** your changes thoroughly
5. **Update** documentation as needed
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### **PR Guidelines**
- Provide clear description of changes
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style

## 🎓 **Research Standards**

For research-related contributions:

- **Cite sources** properly
- **Validate** against known results when possible
- **Provide references** to academic papers
- **Explain** theoretical justification
- **Include** example applications

## 🌱 **Sustainability Focus**

This project promotes regenerative economics principles:

- **Long-term thinking** over short-term gains
- **System health** over individual optimization
- **Resilience** and **adaptability**
- **Collaborative** rather than competitive approaches

## 💬 **Community Guidelines**

- Be respectful and inclusive
- Focus on constructive feedback
- Share knowledge generously
- Support learning and growth
- Maintain professional discourse

## 🏷️ **Issue Labels**

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `research`: Theoretical or methodological contributions
- `visualization`: Charts and UI improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## 📞 **Getting Help**

- Open an issue for questions
- Review existing documentation
- Check the theoretical references
- Participate in discussions

## 🙏 **Acknowledgments**

Contributors will be recognized in:
- README.md acknowledgments section
- Individual commit credits
- Release notes for significant contributions

Thank you for helping make organizational sustainability analysis more accessible and effective!

---

## 📚 **Resources**

### **Key Papers**
- Ulanowicz, R.E. "A Third Window: Natural Life Beyond Newton and Darwin"
- Fath, B.D. & Ulanowicz, R.E. "Measuring Regenerative Economics" (2019)

### **Technical References**
- Information Theory and Network Analysis
- Complex Systems and Emergence
- Organizational Behavior and Systems Thinking

### **Development Tools**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [NetworkX Documentation](https://networkx.org/)

Happy contributing! 🌱