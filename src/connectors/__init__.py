"""Self-provisioning network-source connectors (Gmail first).

Two-stage design: GmailConnector.sync() pulls metadata into GmailInteractionStore;
build_flow_matrix() turns stored rows into a weighted flow matrix for analysis.

Exports are resolved lazily (PEP 562) so importing one submodule does not force
importing its siblings — this keeps each module independently importable while the
package is being built out task-by-task.
"""

__all__ = ["GmailInteractionStore", "build_flow_matrix", "GmailConnector"]


def __getattr__(name):
    if name == "GmailInteractionStore":
        from .gmail_store import GmailInteractionStore
        return GmailInteractionStore
    if name == "build_flow_matrix":
        from .gmail_weighting import build_flow_matrix
        return build_flow_matrix
    if name == "GmailConnector":
        from .gmail_connector import GmailConnector
        return GmailConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
