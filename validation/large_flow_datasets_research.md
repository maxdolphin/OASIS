# Large Flow Datasets for Network Analysis - Research Report

## Executive Summary

This research identifies large datasets from various sources that contain flow data suitable for building flow matrices and applying Ulanowicz information theory metrics. The datasets span supply chain, energy, financial, and material flow domains.

## 🏭 Supply Chain Flow Datasets

### Kaggle Sources

#### 1. **DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS** ⭐⭐⭐⭐⭐
- **Source**: Kaggle (December 2019)
- **URL**: `https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`
- **Size**: Large-scale dataset
- **Why it fits**:
  - Contains comprehensive supply chain network data with multiple nodes
  - Includes delivery performance, customer segments, and product categories
  - Has order priorities and shipping modes creating natural flow pathways
  - Rich enough to construct flow matrices between suppliers, warehouses, and customers
  - Multiple flow types: information, materials, and financial flows

#### 2. **Logistics and Supply Chain Dataset** ⭐⭐⭐⭐
- **Source**: Kaggle (October 2024) - **RECENT**
- **URL**: `https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset`
- **Why it fits**:
  - Most recent supply chain dataset (2024)
  - Contains logistics flow information
  - Suitable for analyzing supply chain network efficiency
  - Can model flows between different logistics nodes

#### 3. **Supply Chain Data** ⭐⭐⭐
- **Source**: Kaggle (February 2022)  
- **URL**: `https://www.kaggle.com/datasets/laurinbrechter/supply-chain-data`
- **Why it fits**:
  - Contains data of producers, ports, and customers
  - Natural network structure with clear flow pathways
  - Can construct matrices showing material flows between entities

## ⚡ Energy Flow Datasets

### Kaggle Sources

#### 1. **European Power Grid Network Dataset** ⭐⭐⭐⭐⭐
- **Source**: Kaggle (March 2024) - **RECENT**
- **URL**: `https://www.kaggle.com/datasets/pythonafroz/european-power-grid-network-dataset`
- **Why it fits**:
  - Specifically designed for power grid network analysis
  - Contains network topology with flow information
  - Perfect for building energy flow matrices
  - European-wide scope provides large-scale network
  - Recent dataset with current grid configurations

#### 2. **Power-Grid Dataset** ⭐⭐⭐⭐
- **Source**: Kaggle (November 2024) - **VERY RECENT**
- **URL**: `https://www.kaggle.com/datasets/ziya07/power-grid`
- **Why it fits**:
  - Most recent power grid dataset (November 2024)
  - Designed for optimization and fault detection
  - Contains flow data suitable for network analysis
  - Can model energy flows between grid nodes

#### 3. **Electrical Grid Power (2015-2021)** ⭐⭐⭐⭐
- **Source**: Kaggle
- **URL**: `https://www.kaggle.com/datasets/l3llff/electrical-grid-power-mw-20152021`
- **Why it fits**:
  - Extensive time series data (2015-2021)
  - 15-minute interval power flow data
  - Large-scale electrical grid measurements
  - Can construct temporal flow matrices

#### 4. **Smart Grid Real-Time Load Monitoring Dataset** ⭐⭐⭐
- **Source**: Kaggle (February 2025) - **FUTURE PROJECTION**
- **URL**: `https://www.kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset`
- **Why it fits**:
  - Real-time load monitoring data
  - Time-series energy flow information
  - Suitable for dynamic flow analysis

## 💰 Financial Flow Datasets

### Kaggle Sources

#### 1. **Synthetic Financial Datasets For Fraud Detection (PaySim)** ⭐⭐⭐⭐⭐
- **Source**: Kaggle
- **URL**: `https://www.kaggle.com/datasets/ealaxi/paysim1`
- **Why it fits**:
  - Large-scale synthetic mobile money transaction data
  - Contains sender-receiver flow information
  - Multiple transaction types creating diverse flow patterns
  - Can construct financial flow matrices between accounts
  - Suitable for studying money flow networks

#### 2. **Financial Transactions Dataset: Analytics** ⭐⭐⭐⭐
- **Source**: Kaggle (October 2024) - **RECENT**
- **URL**: `https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets`
- **Why it fits**:
  - Most recent financial dataset (October 2024)
  - Designed for financial analysis and AI-powered banking
  - Contains transaction flow data
  - Suitable for network analysis of financial flows

#### 3. **BankSim: Synthetic Payment System Data** ⭐⭐⭐⭐
- **Source**: Kaggle
- **URL**: `https://www.kaggle.com/datasets/ealaxi/banksim1`
- **Why it fits**:
  - Synthetic payment system simulation data
  - Models realistic banking transaction flows
  - Contains account-to-account flow information
  - Good for studying payment network structures

## 🏗️ Material Flow Datasets

### Official Sources (OECD/Eurostat)

#### 1. **Eurostat Material Flow Accounts (2024)** ⭐⭐⭐⭐⭐
- **Source**: Eurostat (2024 Edition) - **RECENT**
- **URL**: `https://ec.europa.eu/eurostat/cache/metadata/en/env_ac_mfa_sims.htm`
- **Why it fits**:
  - Official EU material flow data with 99.99% completeness rate
  - 67 categories covering biomass, metals, minerals, fossil fuels
  - International trade flows between countries
  - Perfect for large-scale material flow matrices
  - CSV and Excel formats available

#### 2. **OECD Inter-Country Input-Output Tables (ICIO)** ⭐⭐⭐⭐⭐
- **Source**: OECD (2024)
- **URL**: `https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html`
- **Why it fits**:
  - International statistical infrastructure for flow analysis
  - Maps production, consumption, and trade flows between countries
  - Standard format for input-output analysis
  - Massive scale covering multiple countries and sectors
  - Direct application to flow matrix construction

#### 3. **FIGARO Inter-Country Supply-Use-Input-Output Tables** ⭐⭐⭐⭐⭐
- **Source**: Eurostat (2024 Edition) - **RECENT**
- **URL**: Available through CIRCABC platform
- **Why it fits**:
  - 2024 edition covers 2010-2022 time series
  - EU inter-country supply, use, and input-output tables
  - Perfect for analyzing socioeconomic flows
  - Parquet and CSV formats available
  - Designed specifically for flow analysis

## 🌍 International Trade Flow Datasets

### Official Sources

#### 1. **WTO Trade Statistics (BaTiS)** ⭐⭐⭐⭐⭐
- **Source**: World Trade Organization (2024)
- **URL**: `https://www.wto.org/english/res_e/statis_e/trade_datasets_e.htm`
- **Why it fits**:
  - Complete, consistent, balanced matrix of international trade
  - Covers trade in services globally
  - Bulk download in CSV format
  - Natural network structure (country-to-country flows)
  - Massive scale ($33 trillion global trade in 2024)

#### 2. **CEPII Trade Databases** ⭐⭐⭐⭐⭐
- **Source**: CEPII, France (2024)
- **URL**: `https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele.asp`
- **Why it fits**:
  - TradeProd database: 165 countries, 9 sectors, 1966-2020
  - Network Trade dataset specifically for World Trade Network analysis
  - Designed for network analysis visualization
  - International and domestic trade flows
  - Research-quality data with network analysis focus

#### 3. **UN Comtrade Database** ⭐⭐⭐⭐
- **Source**: United Nations (2024)
- **URL**: `https://comtrade.un.org/`
- **Why it fits**:
  - Comprehensive international trade database
  - Country-to-country commodity flows
  - Large scale and global coverage
  - Standardized trade flow data
  - API access available for bulk downloads

## 🤖 Hugging Face Datasets

### Limited but Growing Selection

#### 1. **Trade Analysis Dataset** ⭐⭐⭐
- **Source**: Hugging Face
- **URL**: `https://huggingface.co/datasets/nguyentranai07/Trade_Analyze_all`
- **Why it fits**:
  - Trade analysis focused
  - Structured format suitable for ML
  - Part of growing ecosystem

#### 2. **E-commerce Flow Data** ⭐⭐
- **Source**: Hugging Face
- **URL**: `https://huggingface.co/datasets/TrainingDataPro/asos-e-commerce-dataset`
- **Why it fits**:
  - 30,845+ items with transaction-like structure
  - Can model customer-product flows
  - Good for retail network analysis

## 📊 Dataset Evaluation Criteria

### Size and Scale ⚖️
- **Large**: >100K records or entities
- **Medium**: 10K-100K records
- **Small**: <10K records

### Flow Matrix Suitability 🔄
- **Excellent**: Direct source-destination flows
- **Good**: Can derive flows from relationships
- **Fair**: Requires significant preprocessing

### Data Quality 📈
- **High**: Official/research institutions, validated data
- **Medium**: Commercial/crowdsourced, some validation
- **Low**: Unvalidated or synthetic data

### Recency 📅
- **2024-2025**: Most current data
- **2020-2023**: Recent and relevant
- **Pre-2020**: Historical but potentially outdated

## 🎯 Top Recommendations

### For Network Flow Matrix Construction:

1. **OECD Inter-Country Input-Output Tables** - Perfect structure, official data
2. **Eurostat FIGARO Tables (2024)** - Recent, comprehensive, EU-wide
3. **European Power Grid Network Dataset** - Energy flows, recent, network-ready
4. **DataCo Smart Supply Chain** - Complete supply chain network
5. **PaySim Financial Dataset** - Large-scale financial flows

### For Testing Our Implementation:

1. **WTO Trade Statistics** - Massive scale, validated flows
2. **CEPII Network Trade Dataset** - Specifically designed for network analysis  
3. **Eurostat Material Flow Accounts** - Official, comprehensive, recent

## 🔧 Implementation Strategy

### Phase 1: Validation
- Start with **European Power Grid** and **DataCo Supply Chain** datasets
- Verify our Ulanowicz calculations against published benchmarks
- Build confidence in flow matrix construction

### Phase 2: Scale Testing
- Apply to **OECD ICIO** and **FIGARO** datasets
- Test computational performance on large matrices
- Validate against economic literature

### Phase 3: Domain Expansion  
- Integrate **financial flow** datasets for money networks
- Add **material flow** datasets for sustainability analysis
- Develop **trade flow** visualizations for global networks

## 📋 Next Steps

1. **Download and explore** top 3 recommended datasets
2. **Build data loaders** for each dataset type
3. **Create flow matrix converters** from raw data
4. **Validate calculations** against known benchmarks
5. **Develop visualization templates** for each domain
6. **Document data sources** and preprocessing steps

## 🚨 Data Quality Considerations

- **Eurostat/OECD**: Highest quality, official validation
- **Kaggle**: Variable quality, check data provenance
- **Synthetic datasets**: Good for testing, limited real-world applicability
- **Recent datasets (2024)**: Prefer for current relevance
- **Large scale**: Essential for meaningful network analysis

---

*Research compiled on January 27, 2025 - Datasets identified for flow matrix construction and Ulanowicz information theory analysis*