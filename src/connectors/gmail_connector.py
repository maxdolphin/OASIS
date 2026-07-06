"""Stage 1: Gmail metadata sync into GmailInteractionStore.

GmailConnector subclasses BaseConnector. Admin/Gmail clients are injected so the
logic is testable with fakes; production builds real clients from credentials.
Only metadata headers are read (From/To/Cc/timestamp/thread/size) — never body.
"""
from __future__ import annotations

from datetime import datetime
from email.utils import getaddresses
from typing import Any, Dict, Optional

import numpy as np

try:
    from cloud_connectors import BaseConnector
except ImportError:  # pragma: no cover
    from src.cloud_connectors import BaseConnector

from .gmail_store import GmailInteractionStore
from .gmail_weighting import build_flow_matrix

# Read-only, least-privilege scopes (metadata scope cannot read body/subject).
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/admin.directory.user.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
]


class GmailConnector(BaseConnector):
    def __init__(self, admin_client=None, gmail_client=None,
                 domain: Optional[str] = None,
                 store: Optional[GmailInteractionStore] = None):
        self.admin_client = admin_client
        self.gmail_client = gmail_client
        self.domain = domain
        self.store = store or GmailInteractionStore()

    # --- BaseConnector contract -------------------------------------------------

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Build Admin + Gmail clients from a service-account credential dict.

        Required keys: 'service_account_file', 'subject' (admin to impersonate),
        'domain'. Returns False (never raises) on any missing key or build error.
        """
        required = ("service_account_file", "subject", "domain")
        if not all(credentials.get(k) for k in required):
            return False
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                credentials["service_account_file"], scopes=GMAIL_SCOPES,
            ).with_subject(credentials["subject"])
            admin = _AdminSdkClient(build("admin", "directory_v1", credentials=creds))
            gmail = _GmailApiClient(credentials, GMAIL_SCOPES)
            self.admin_client, self.gmail_client = admin, gmail
            self.domain = credentials["domain"]
            return True
        except Exception as exc:  # pragma: no cover - real-API path
            print(f"Gmail authenticate failed: {exc}")
            return False

    def get_organization_structure(self) -> Dict[str, Any]:
        users = self.admin_client.list_users()
        user_orgunit = {u["primaryEmail"].lower(): u.get("orgUnitPath", "/")
                        for u in users}
        return {
            "org_users": set(user_orgunit.keys()),
            "user_orgunit": user_orgunit,
            "total_users": len(user_orgunit),
        }

    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """BaseConnector convenience: sync the window then build with defaults."""
        start_ts, end_ts = int(start_date.timestamp()), int(end_date.timestamp())
        self.sync(start_ts, end_ts, sync_run_id=f"flowdata-{start_ts}")
        org = self.get_organization_structure()
        rows = self.store.query_window(self.domain, start_ts, end_ts)
        parsed, _ = build_flow_matrix(
            rows, org_users=org["org_users"], now_utc=end_ts,
            window_seconds=end_ts - start_ts, half_life_seconds=30 * 86400,
            beta=0.5, granularity="individual")
        return parsed.flow_matrix

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "connector": "Gmail",
            "domain": self.domain,
            "data_sources": ["Gmail metadata", "Admin SDK"],
            "privacy": "metadata-only (From/To/Cc/timestamp/thread/size)",
        }

    # --- Stage 1: sync ----------------------------------------------------------

    def sync(self, start_ts: int, end_ts: int, sync_run_id: str) -> int:
        """Pull metadata for every org user in [start_ts, end_ts]; store rows.

        Returns the number of directed rows written (one per To/Cc recipient).
        """
        org = self.get_organization_structure()
        user_orgunit = org["user_orgunit"]
        rows = []
        for sender in user_orgunit:
            for msg in self.gmail_client.list_sent_messages(sender, start_ts, end_ts):
                src = str(msg["from"]).lower()
                for kind in ("to", "cc"):
                    for rcpt in msg.get(kind, []) or []:
                        dst = str(rcpt).lower()
                        rows.append({
                            "src_email": src, "dst_email": dst,
                            "recipient_kind": kind, "ts_utc": int(msg["ts_utc"]),
                            "thread_id": msg.get("thread_id"),
                            "size_bytes": msg.get("size_bytes"),
                            "src_orgunit": user_orgunit.get(src),
                            "dst_orgunit": user_orgunit.get(dst),
                        })
        if not rows:
            return 0
        return self.store.insert_rows(self.domain, sync_run_id, rows)


class _AdminSdkClient:  # pragma: no cover - thin real-API adapter
    """Adapts the googleapiclient Admin SDK to the list_users() duck type."""
    def __init__(self, service):
        self.service = service

    def list_users(self):
        out, page = [], None
        while True:
            resp = self.service.users().list(
                customer="my_customer", maxResults=500, pageToken=page,
                projection="full").execute()
            out.extend(resp.get("users", []))
            page = resp.get("nextPageToken")
            if not page:
                break
        return out


class _GmailApiClient:  # pragma: no cover - real-API adapter, exercised via fakes in tests
    """Adapts the Gmail API to list_sent_messages(user, start_ts, end_ts).

    Uses format='metadata' so message bodies are never fetched.
    """
    def __init__(self, credentials: Dict[str, Any], scopes):
        self._credentials = credentials
        self._scopes = scopes

    def _service_for(self, user_email):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_file(
            self._credentials["service_account_file"], scopes=self._scopes,
        ).with_subject(user_email)
        return build("gmail", "v1", credentials=creds)

    def list_sent_messages(self, user_email, start_ts, end_ts):
        service = self._service_for(user_email)
        query = f"in:sent after:{start_ts} before:{end_ts}"
        results = []
        page = None
        while True:
            resp = service.users().messages().list(
                userId="me", q=query, pageToken=page).execute()
            for ref in resp.get("messages", []):
                msg = service.users().messages().get(
                    userId="me", id=ref["id"], format="metadata",
                    metadataHeaders=["From", "To", "Cc", "Date"]).execute()
                results.append(_parse_metadata(msg))
            page = resp.get("nextPageToken")
            if not page:
                break
        return results


def _parse_metadata(msg) -> Dict[str, Any]:
    headers = {h["name"].lower(): h["value"]
               for h in msg.get("payload", {}).get("headers", [])}

    def addrs(v):
        return [addr for _, addr in getaddresses([v]) if addr] if v else []

    return {
        "ts_utc": int(msg.get("internalDate", "0")) // 1000,
        "thread_id": msg.get("threadId"),
        "size_bytes": msg.get("sizeEstimate"),
        "from": headers.get("from", ""),
        "to": addrs(headers.get("to", "")),
        "cc": addrs(headers.get("cc", "")),
    }
