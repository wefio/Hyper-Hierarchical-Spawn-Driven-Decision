"""L2Cache — peer-shared in-process cache (CPU L2 analogy).

Each spawn group gets one L2Cache instance. Siblings publish step results
as pointers (summary + ptr_id); siblings read each other's pointer lists.
Zero-trust: entries are hash-signed with the namespace_id. Disk backup is
snapshotted with a signature file; restore rejects tampered snapshots.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

from config import Config


class L2Cache:
    """Shared cache for sibling agents within one spawn group.

    Lifecycle:
        parent creates → injects to all children → children publish step
        results → children read peer pointers → parent destroys on completion.

    Storage:
        primary: in-memory dict
        backup:  snapshot file (.bak) + signature (.sig), overwrite on write
    """

    def __init__(self, namespace_id: str, parent_scope: str):
        self.namespace_id = namespace_id
        self.parent_scope = parent_scope
        self._data: dict[str, dict] = {}  # key: "agent_id:step_id"
        self._lock = threading.RLock()

        self._backup_dir = Path(Config.WORK_DIR) / "l2_snapshots"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._bak_path = self._backup_dir / f"{namespace_id}.bak"
        self._sig_path = self._backup_dir / f"{namespace_id}.sig"

    # ---- signing ----

    def _sign(self, data: str) -> str:
        """Sign data with namespace_id as HMAC key."""
        return hashlib.sha256(
            (data + self.namespace_id).encode("utf-8")
        ).hexdigest()

    def _make_sig(self, entry: dict) -> str:
        """Compute signature for a single entry's content fields."""
        raw = f"{entry['agent_id']}:{entry['step_id']}:{entry['summary']}:{entry['ptr_id']}:{entry['timestamp']}"
        return self._sign(raw)

    # ---- write ----

    def publish(self, agent_id: str, step_id: int, summary: str,
                ptr_id: str, tokens: int = 0):
        """Publish a step result pointer to L2 (called by engine after step)."""
        entry = {
            "agent_id": agent_id,
            "step_id": step_id,
            "summary": summary[:300],
            "ptr_id": ptr_id,
            "tokens": tokens,
            "timestamp": time.time(),
        }
        entry["sig"] = self._make_sig(entry)

        with self._lock:
            key = f"{agent_id}:{step_id}"
            self._data[key] = entry

    # ---- read ----

    def read_peer(self, agent_id: str = None,
                  step_id: int = None,
                  query: str = None) -> list[dict]:
        """Return peer pointer list. Gatekeeper: implicit namespace via object ref.

        Returns entries matching optional filters. All entries are signature-
        verified; tampered entries are silently dropped.
        """
        with self._lock:
            entries = list(self._data.values())

        result = []
        for e in entries:
            # verify signature
            expected = self._make_sig({
                "agent_id": e["agent_id"], "step_id": e["step_id"],
                "summary": e["summary"], "ptr_id": e["ptr_id"],
                "timestamp": e["timestamp"],
            })
            if expected != e.get("sig", ""):
                continue  # tampered, drop silently

            if agent_id and e["agent_id"] != agent_id:
                continue
            if step_id is not None and e["step_id"] != step_id:
                continue
            if query and query.lower() not in e.get("summary", "").lower():
                continue
            result.append({
                "agent_id": e["agent_id"],
                "step_id": e["step_id"],
                "summary": e["summary"],
                "ptr_id": e["ptr_id"],
                "tokens": e["tokens"],
            })

        return result

    # ---- disk backup ----

    def snapshot(self) -> None:
        """Write full L2 state to disk with signature. Overwrites previous."""
        with self._lock:
            payload = {
                "namespace_id": self.namespace_id,
                "parent_scope": self.parent_scope,
                "entries": list(self._data.values()),
            }
            raw = json.dumps(payload, ensure_ascii=False)
            sig = self._sign(raw)
        try:
            self._bak_path.write_text(raw, encoding="utf-8")
            self._sig_path.write_text(sig, encoding="utf-8")
        except OSError:
            pass  # backup is best-effort, not critical

    @classmethod
    def restore(cls, namespace_id: str,
                backup_dir: str = None) -> Optional["L2Cache"]:
        """Restore L2 from disk backup. Returns None if tampered or missing."""
        base = Path(backup_dir or (Path(Config.WORK_DIR) / "l2_snapshots"))
        bak = base / f"{namespace_id}.bak"
        sig_file = base / f"{namespace_id}.sig"
        if not bak.exists() or not sig_file.exists():
            return None
        try:
            raw = bak.read_text(encoding="utf-8")
            saved_sig = sig_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None

        # verify
        expected = hashlib.sha256(
            (raw + namespace_id).encode("utf-8")
        ).hexdigest()
        if expected != saved_sig:
            # tampered — discard
            try:
                bak.unlink()
                sig_file.unlink()
            except OSError:
                pass
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        cache = cls(namespace_id, payload.get("parent_scope", ""))
        with cache._lock:
            for e in payload.get("entries", []):
                key = f"{e.get('agent_id', '')}:{e.get('step_id', 0)}"
                cache._data[key] = e
        return cache

    # ---- lifecycle ----

    def destroy(self):
        """Destroy L2: snapshot one last time, then clear memory. Backup files kept."""
        self.snapshot()
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
