from dataclasses import dataclass

ACTIVE = "active"
ARCHIVE = "archive"


@dataclass
class PolicyConfig:
    active_budget: int = 40         # max active memories per session before eviction kicks in
    grace_turns: int = 3            # never evict something accessed in the last N turns
    half_life_turns: float = 20.0   # recency score halves every N turns without access
    w_recency: float = 0.4
    w_frequency: float = 0.2
    w_used: float = 0.4
    top_k: int = 5                  # memories returned per recall
    archive_k: int = 3              # archive candidates checked for reload
    min_similarity: float = 0.6     # active hits below this are ignored (bge scores unrelated text ~0.4-0.5)
    reload_threshold: float = 0.65  # archive hits above this get pulled back to active (tuned by eval sweep)
    used_threshold: float = 0.2     # lexical overlap needed to count a memory as "used"
    recent_window: int = 4          # raw recent messages sent alongside memories


@dataclass
class MemoryRecord:
    id: str
    session_id: str
    text: str
    tier: str
    created_turn: int
    last_access_turn: int
    retrieved_count: int = 0
    used_count: int = 0


@dataclass
class RecallHit:
    memory: MemoryRecord
    similarity: float
    reloaded: bool = False
