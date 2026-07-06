"""Self-provisioning network-source connectors (Gmail first).

Two-stage design: GmailConnector.sync() pulls metadata into GmailInteractionStore;
build_flow_matrix() turns stored rows into a weighted flow matrix for analysis.
"""

from .gmail_store import GmailInteractionStore
from .gmail_weighting import build_flow_matrix
from .gmail_connector import GmailConnector

__all__ = ["GmailInteractionStore", "build_flow_matrix", "GmailConnector"]
