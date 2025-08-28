# 🎭 Roles Analysis Implementation Plan
## Based on Zorach & Ulanowicz (2003): "Quantifying the Complexity of Flow Networks"

---

## 📋 Executive Summary

Implement network roles analysis to quantify functional specialization and complexity in organizational/ecosystem networks. This feature measures the number of distinct functional roles (specialized functions) in a network, providing insights into system differentiation and complexity.

**Key Finding:** AMI (Average Mutual Information) is already calculated in our system, and **R = exp(AMI)** gives us the number of roles directly!

---

## 🔬 Scientific Foundation

### Core Concepts from Zorach & Ulanowicz (2003)

1. **Role Definition**: A specialized function where nodes take inputs from specific sources and pass them to specific destinations
2. **Geometric Mean**: The only mathematically consistent mean for preserving role relationships
3. **Information Theory Connection**: log(R) = AMI (Average Mutual Information)
4. **Natural Systems Window**: Ecosystems exist in narrow parameter space (C ∈ [1, 3.25], R ∈ [2, 5])

### Mathematical Formulas

```
Effective Flows:         F = ∏((Tij/T••)^(-Tij/T••))
Effective Nodes:         N = ∏((T••²/(Ti•*T•j))^(1/2*Tij/T••))
Effective Connectivity:  C = ∏((Tij²/(Ti•*T•j))^(1/2*Tij/T••))
Number of Roles:         R = ∏((Tij*T••/(Ti•*T•j))^(Tij/T••))

Key Relationships:
- R = exp(AMI)
- R = N²/F = F/C² = N/C
- log(R) = AMI (Average Mutual Information)
```

---

## ✅ Existing Framework Analysis

### Already Implemented and Working:

| Component | Location | Status | Notes |
|-----------|----------|---------|-------|
| AMI Calculation | `ulanowicz_calculator.py:82-94` | ✅ Working | Exact formula needed for roles |
| TST (Total System Throughput) | `ulanowicz_calculator.py:79` | ✅ Working | Foundation metric |
| Flow Matrix Components | Throughout | ✅ Working | Tij, Ti•, T•j all calculated |
| Flow Diversity (H) | `ulanowicz_calculator.py:473` | ✅ Working | Needed for F calculation |
| Conditional Entropy | `ulanowicz_calculator.py:509` | ✅ Working | Related to overhead |
| Overhead (Φ) | `ulanowicz_calculator.py:251` | ✅ Working | Related to C calculation |

### Referenced but NOT Implemented:

- **Functional Diversity (Roles)** mentioned in `app.py:5967` but not calculated
- Formula shown but no actual implementation exists

### Current Display Hierarchy:

1. **Level 1:** Data & Flow Statistics
2. **Level 2:** Network Topology
3. **Level 3:** Ulanowicz Core Metrics
4. **[NEW] Level 4:** Network Roles & Specialization ← Insert here
5. **Mathematical Validation**

---

## 🏗️ Implementation Plan

### Phase 1: Calculator Updates
**File:** `src/ulanowicz_calculator.py`

#### 1.1 Add Core Calculation Methods

```python
def calculate_effective_flows(self) -> float:
    """
    Calculate effective number of flows (F).
    F = exp(-H) where H is flow diversity (Shannon entropy of flows).
    """
    flow_diversity = self.calculate_flow_diversity()
    return np.exp(-flow_diversity)

def calculate_effective_nodes(self) -> float:
    """
    Calculate effective number of nodes (N).
    Based on normalized throughput distribution.
    """
    tst = self.calculate_tst()
    if tst == 0:
        return self.n_nodes
    
    sum_term = 0
    for i in range(self.n_nodes):
        for j in range(self.n_nodes):
            if self.flow_matrix[i, j] > 0:
                tij = self.flow_matrix[i, j]
                ti_out = np.sum(self.flow_matrix[i, :])
                tj_in = np.sum(self.flow_matrix[:, j])
                
                if ti_out > 0 and tj_in > 0:
                    weight = tij / tst
                    sum_term += weight * np.log(tst**2 / (ti_out * tj_in))
    
    return np.exp(0.5 * sum_term)

def calculate_effective_connectivity(self) -> float:
    """
    Calculate effective connectivity (C).
    C = sqrt(F * R) based on relationships in paper.
    """
    eff_flows = self.calculate_effective_flows()
    num_roles = self.calculate_number_of_roles()
    return np.sqrt(eff_flows * num_roles)

def calculate_number_of_roles(self) -> float:
    """
    Calculate number of functional roles (R).
    R = exp(AMI) - direct from Zorach & Ulanowicz (2003).
    """
    ami = self.calculate_ami()
    return np.exp(ami)

def calculate_roles_metrics(self) -> Dict[str, float]:
    """
    Calculate all roles-related metrics.
    Returns comprehensive roles analysis.
    """
    # Calculate base metrics
    num_roles = self.calculate_number_of_roles()
    eff_nodes = self.calculate_effective_nodes()
    eff_flows = self.calculate_effective_flows()
    eff_connectivity = self.calculate_effective_connectivity()
    
    # Verification checks
    verification1 = num_roles - (eff_nodes**2 / eff_flows)
    verification2 = num_roles - (eff_flows / eff_connectivity**2)
    verification3 = num_roles - (eff_nodes / eff_connectivity)
    
    return {
        'number_of_roles': num_roles,
        'effective_nodes': eff_nodes,
        'effective_flows': eff_flows,
        'effective_connectivity': eff_connectivity,
        'roles_per_node': num_roles / eff_nodes if eff_nodes > 0 else 0,
        'specialization_index': num_roles / self.n_nodes,
        'functional_diversity': np.log(num_roles),  # This is AMI
        'roles_verification_error': max(abs(verification1), abs(verification2), abs(verification3))
    }
```

#### 1.2 Update get_extended_metrics()

Add to the extended_metrics dictionary (around line 724):

```python
# Add roles analysis
roles_metrics = self.calculate_roles_metrics()
extended_metrics.update(roles_metrics)
```

---

### Phase 2: Display Integration
**File:** `app.py`

#### 2.1 Add Roles Analysis Section (insert at line ~2375, before Mathematical Validation)

```python
# Network Roles & Functional Specialization
st.markdown("---")
st.subheader("🎭 Level 4: Network Roles & Functional Specialization")
st.markdown("*Based on Zorach & Ulanowicz (2003) - Quantifying the complexity of flow networks*")

# Core roles metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of Roles", f"{metrics.get('number_of_roles', 0):.2f}")
    st.caption("R = exp(AMI) [roles]")
    
with col2:
    st.metric("Effective Nodes", f"{metrics.get('effective_nodes', 0):.2f}")
    st.caption("N = weighted nodes [nodes]")
    
with col3:
    st.metric("Effective Flows", f"{metrics.get('effective_flows', 0):.2f}")
    st.caption("F = weighted flows [flows]")
    
with col4:
    st.metric("Effective Connectivity", f"{metrics.get('effective_connectivity', 0):.2f}")
    st.caption("C = F/N [flows/node]")

# Interpretation metrics
st.markdown("#### 🔍 Specialization Analysis")
col1, col2, col3, col4 = st.columns(4)

with col1:
    roles_per_node = metrics.get('roles_per_node', 0)
    st.metric("Roles per Node", f"{roles_per_node:.3f}")
    st.caption("R/N [roles/node]")
    
with col2:
    spec_index = metrics.get('specialization_index', 0)
    st.metric("Specialization Index", f"{spec_index:.3f}")
    st.caption("R/N_actual [dimensionless]")
    
with col3:
    # Compare actual vs effective
    actual_nodes = len(node_names)
    node_ratio = metrics.get('effective_nodes', 0) / actual_nodes if actual_nodes > 0 else 0
    st.metric("Node Utilization", f"{node_ratio:.2%}")
    st.caption("N_eff/N_actual [%]")
    
with col4:
    # Verification
    verif_error = metrics.get('roles_verification_error', 0)
    if verif_error < 0.01:
        st.metric("Math Check", "✅ Valid")
    else:
        st.metric("Math Check", f"⚠️ {verif_error:.4f}")
    st.caption("R = N²/F = F/C² check")

# Assessment based on roles
num_roles = metrics.get('number_of_roles', 0)
if num_roles < 2:
    assessment = "⚠️ **Low Specialization**: System lacks functional differentiation"
    color = "warning"
elif 2 <= num_roles <= 5:
    assessment = "✅ **Optimal Specialization**: Natural range for sustainable systems"
    color = "success"
else:
    assessment = "⚠️ **Over-Specialized**: System may be brittle or overly complex"
    color = "warning"

st.info(assessment)

# Add small visualization if feasible
if num_roles <= 10:
    # Create simple bar chart comparing actual vs effective
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(name='Actual', x=['Nodes', 'Flows'], 
               y=[len(node_names), np.count_nonzero(flow_matrix)]),
        go.Bar(name='Effective', x=['Nodes', 'Flows'],
               y=[metrics.get('effective_nodes', 0), metrics.get('effective_flows', 0)])
    ])
    fig.update_layout(
        title="Actual vs Effective Network Components",
        barmode='group',
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
```

---

### Phase 3: Update Formulas Reference
**File:** `app.py` (Formulas Reference section ~line 5965)

Update the existing Functional Diversity section with proper implementation details:

```python
### **6. Functional Diversity (Roles) - Zorach & Ulanowicz (2003)**

The number of functional roles quantifies system complexity and specialization:

```
Number of Roles:    R = exp(AMI) = ∏((Tij*T••/(Ti•*T•j))^(Tij/T••))
Effective Nodes:    N = ∏((T••²/(Ti•*T•j))^(1/2*Tij/T••))  
Effective Flows:    F = ∏((Tij/T••)^(-Tij/T••))
Effective Connect:  C = ∏((Tij²/(Ti•*T•j))^(1/2*Tij/T••))

Fundamental Relationships:
- R = N²/F = F/C² = N/C
- log(R) = AMI (Average Mutual Information)
- R measures degree of functional specialization
```

**Interpretation:**
- **R < 2**: Undifferentiated system, all nodes perform similar functions
- **2 ≤ R ≤ 5**: Natural range for sustainable ecosystems  
- **R > 5**: Over-specialized, potentially brittle system

**Applications:**
- Organizational structure analysis
- Ecosystem complexity assessment  
- Supply chain specialization evaluation
- Neural network functional diversity
```

---

## 📊 Testing & Validation

### Test Cases:

1. **Mathematical Consistency**
   - Verify R = N²/F = F/C² = N/C (error < 0.01)
   - Confirm log(R) = AMI

2. **Ecosystem Samples**
   - Coral Reef: Expect R ∈ [3, 5]
   - Cypress Wetlands: Expect R ∈ [2, 4]
   - Crystal River: Expect R ∈ [2, 4]

3. **Edge Cases**
   - Empty matrix: R = 0
   - Single flow: R = 1
   - Fully connected: R approaches N

4. **Performance**
   - Networks up to 100 nodes should compute in < 1 second
   - No timeout issues like Finn Cycling Index

---

## 🎯 Success Criteria

✅ **Technical Requirements:**
- [ ] All roles metrics calculate correctly
- [ ] Mathematical relationships verified (R = N²/F etc.)
- [ ] Integration with existing metrics seamless
- [ ] No breaking changes to existing code

✅ **User Experience:**
- [ ] Clear visual presentation of roles metrics
- [ ] Intuitive interpretation guidance
- [ ] Proper positioning in computation flow
- [ ] Helpful tooltips and explanations

✅ **Scientific Accuracy:**
- [ ] Formulas match Zorach & Ulanowicz (2003) exactly
- [ ] Natural systems show expected R ∈ [2, 5] range
- [ ] Random networks show broader distribution
- [ ] Proper citation and references

---

## 📚 References

**Primary Source:**
Zorach, A. C., & Ulanowicz, R. E. (2003). Quantifying the complexity of flow networks: How many roles are there? Complexity, 8(3), 68-76.

**Related Work:**
- Ulanowicz, R. E. (1986). Growth and Development: Ecosystem Phenomenology
- Ulanowicz, R. E. (2002). The balance between adaptability and adaptation

---

## 🚀 Implementation Priority

**High Priority (Core Functionality):**
1. Calculator methods for R, N, F, C
2. Basic display in Core Metrics
3. Mathematical verification

**Medium Priority (Enhanced Features):**
1. Visualization of actual vs effective
2. Detailed interpretations
3. Comparative analysis with benchmarks

**Low Priority (Future Enhancements):**
1. Interactive parameter exploration
2. Role clustering visualization
3. Time series analysis of roles

---

## 📝 Notes

- AMI is already calculated correctly, making R = exp(AMI) straightforward
- All required flow matrix components are available
- Framework is well-suited for this addition
- No computational bottlenecks expected (unlike Finn Cycling Index)
- Clean integration with existing Level 1-3 metrics

---

*Plan created: 2025-08-28*
*Based on: Zorach & Ulanowicz (2003) paper analysis*
*Target implementation: Immediate*