"""SQLite is the source of truth for memory text, tier, and usage stats.
Chroma only holds vectors (plus tier/session metadata for filtered search)."""

import json
import sqlite3
import threading
from dataclasses import astuple

from .models import MemoryRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, turn INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL, tier TEXT NOT NULL,
    created_turn INTEGER NOT NULL, last_access_turn INTEGER NOT NULL,
    retrieved_count INTEGER NOT NULL DEFAULT 0, used_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id, tier);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ops (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, turn INTEGER NOT NULL,
    op TEXT NOT NULL, memory_id TEXT, score REAL, detail TEXT,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

COLS = "id, session_id, text, tier, created_turn, last_access_turn, retrieved_count, used_count"


class Store:
    def __init__(self, path: str = "lethe.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql, params=()):
        with self.lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur

    # --- turns ---
    def current_turn(self, session_id: str) -> int:
        row = self._exec("SELECT turn FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else 0

    def next_turn(self, session_id: str) -> int:
        self._exec(
            "INSERT INTO sessions(session_id, turn) VALUES (?, 1) "
            "ON CONFLICT(session_id) DO UPDATE SET turn = turn + 1",
            (session_id,),
        )
        return self.current_turn(session_id)

    # --- memories ---
    def insert(self, m: MemoryRecord):
        self._exec(f"INSERT INTO memories({COLS}) VALUES (?,?,?,?,?,?,?,?)", astuple(m))

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._exec(f"SELECT {COLS} FROM memories WHERE id=?", (memory_id,)).fetchone()
        return MemoryRecord(*row) if row else None

    def list_memories(self, session_id: str, tier: str | None = None) -> list[MemoryRecord]:
        if tier:
            rows = self._exec(
                f"SELECT {COLS} FROM memories WHERE session_id=? AND tier=?", (session_id, tier)
            ).fetchall()
        else:
            rows = self._exec(f"SELECT {COLS} FROM memories WHERE session_id=?", (session_id,)).fetchall()
        return [MemoryRecord(*r) for r in rows]

    def touch(self, memory_id: str):
        """Count a retrieval. Does NOT refresh recency: being retrieved isn't evidence of value."""
        self._exec("UPDATE memories SET retrieved_count = retrieved_count + 1 WHERE id=?", (memory_id,))

    def mark_used(self, memory_id: str, turn: int):
        """Being used in an answer is what keeps a memory fresh."""
        self._exec(
            "UPDATE memories SET used_count = used_count + 1, last_access_turn=? WHERE id=?", (turn, memory_id)
        )

    def set_tier(self, memory_id: str, tier: str, turn: int | None = None):
        if turn is None:
            self._exec("UPDATE memories SET tier=? WHERE id=?", (tier, memory_id))
        else:
            self._exec("UPDATE memories SET tier=?, last_access_turn=? WHERE id=?", (tier, turn, memory_id))

    # --- recent messages (short-term window) ---
    def add_message(self, session_id: str, role: str, content: str):
        self._exec("INSERT INTO messages(session_id, role, content) VALUES (?,?,?)", (session_id, role, content))

    def recent_messages(self, session_id: str, n: int) -> list[dict]:
        rows = self._exec(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, n)
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # --- ops log ---
    def log(self, session_id: str, turn: int, op: str, memory_id=None, score=None, detail=None):
        self._exec(
            "INSERT INTO ops(session_id, turn, op, memory_id, score, detail) VALUES (?,?,?,?,?,?)",
            (session_id, turn, op, memory_id, score, json.dumps(detail) if detail else None),
        )

    def ops(self, session_id: str, limit: int = 200) -> list[dict]:
        rows = self._exec(
            "SELECT turn, op, memory_id, score, detail, ts FROM ops WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {"turn": t, "op": o, "memory_id": m, "score": s, "detail": json.loads(d) if d else None, "ts": ts}
            for t, o, m, s, d, ts in rows
        ]
