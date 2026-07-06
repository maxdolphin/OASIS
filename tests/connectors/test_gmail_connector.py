from src.connectors.gmail_connector import GmailConnector


class FakeAdmin:
    """Mimics the two calls the connector makes on the Admin SDK."""
    def list_users(self):
        return [
            {"primaryEmail": "a@x.com", "orgUnitPath": "/Sales"},
            {"primaryEmail": "b@x.com", "orgUnitPath": "/IT"},
        ]


class FakeGmail:
    """Returns metadata headers for one sent message with To + Cc."""
    def list_sent_messages(self, user_email, start_ts, end_ts):
        if user_email != "a@x.com":
            return []
        return [{
            "ts_utc": (start_ts + end_ts) // 2,
            "thread_id": "thread-1",
            "size_bytes": 500,
            "from": "a@x.com",
            "to": ["b@x.com"],
            "cc": ["b@x.com"],
        }]


def test_get_organization_structure_maps_users_to_orgunits():
    c = GmailConnector(admin_client=FakeAdmin(), gmail_client=FakeGmail(),
                       domain="x.com")
    org = c.get_organization_structure()
    assert org["user_orgunit"]["a@x.com"] == "/Sales"
    assert org["org_users"] == {"a@x.com", "b@x.com"}


def test_sync_emits_per_recipient_rows(tmp_path):
    from src.connectors.gmail_store import GmailInteractionStore
    store = GmailInteractionStore(db_path=str(tmp_path / "t.db"))
    c = GmailConnector(admin_client=FakeAdmin(), gmail_client=FakeGmail(),
                       domain="x.com", store=store)
    n = c.sync(start_ts=1000, end_ts=3000, sync_run_id="run1")
    assert n == 2  # one To row + one Cc row
    rows = store.query_window("x.com", 0, 10 ** 12)
    kinds = sorted(r["recipient_kind"] for r in rows)
    assert kinds == ["cc", "to"]
    assert all(r["src_email"] == "a@x.com" and r["dst_email"] == "b@x.com"
               for r in rows)
    assert rows[0]["src_orgunit"] == "/Sales"
    assert rows[0]["dst_orgunit"] == "/IT"


def test_authenticate_returns_false_on_missing_credentials():
    c = GmailConnector()
    assert c.authenticate({}) is False
