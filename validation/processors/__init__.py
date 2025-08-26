"""
Real-World Dataset Processors

This module contains processors for converting real-world datasets from various sources
into standardized flow matrices compatible with the Adaptive Organization Analysis System.
"""

from .base_processor import BaseDatasetProcessor
from .energy_processor import EnergyFlowProcessor, EuropeanPowerGridProcessor, SmartGridProcessor
from .supply_chain_processor import SupplyChainProcessor, DataCoSupplyChainProcessor, LogisticsSupplyChainProcessor
from .financial_processor import FinancialFlowProcessor, PaySimFinancialProcessor, BankingTransactionsProcessor
from .official_processor import OfficialDataProcessor, OECDInputOutputProcessor, EurostatMaterialFlowProcessor, WTOTradeProcessor

__all__ = [
    'BaseDatasetProcessor',
    'EnergyFlowProcessor', 'EuropeanPowerGridProcessor', 'SmartGridProcessor',
    'SupplyChainProcessor', 'DataCoSupplyChainProcessor', 'LogisticsSupplyChainProcessor',
    'FinancialFlowProcessor', 'PaySimFinancialProcessor', 'BankingTransactionsProcessor',
    'OfficialDataProcessor', 'OECDInputOutputProcessor', 'EurostatMaterialFlowProcessor', 'WTOTradeProcessor'
]