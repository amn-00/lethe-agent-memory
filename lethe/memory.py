import uuid

from .index import Embedder, VectorIndex
from .models import ACTIVE, ARCHIVE, MemoryRecord, PolicyConfig, RecallHit
from .policy import answer_overlap, retention_score
from .store import Store


class AgentMemory:
    def __init__(self, store: Store, index: VectorIndex, embedder: Embedder, cfg: PolicyConfig | None = None):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.cfg = cfg or PolicyConfig()

    def add(self, session_id: str, text: str, turn: int | None = None) -> MemoryRecord:
        """turn: when the memory was said. Defaults to now; extracted facts pass their source turn."""
        turn = self.store.current_turn(session_id) if turn is None else turn
        m = MemoryRecord(uuid.uuid4().hex, session_id, text, ACTIVE, turn, turn)
        self.store.insert(m)
        self.index.add(m.id, self.embedder.embed([text])[0], session_id, ACTIVE)
        self.store.log(session_id, turn, "add", m.id, detail={"text": text[:120]})
        return m

    def recall(self, session_id: str, query: str) -> list[RecallHit]:
        cfg, turn = self.cfg, self.store.current_turn(session_id)
        emb = self.embedder.embed([query])[0]

        candidates = [
            (mid, sim, False)
            for mid, sim in self.index.query(emb, session_id, ACTIVE, cfg.top_k)
            if sim >= cfg.min_similarity
        ]
        candidates += [
            (mid, sim, True)
            for mid, sim in self.index.query(emb, session_id, ARCHIVE, cfg.archive_k)
            if sim >= cfg.reload_threshold
        ]
        candidates.sort(key=lambda c: c[1], reverse=True)

        hits = []
        for mid, sim, from_archive in candidates[: cfg.top_k]:
            if from_archive:
                self._move(session_id, mid, ACTIVE, turn)
                self.store.log(session_id, turn, "reload", mid, score=sim)
            self.store.touch(mid)
            self.store.log(session_id, turn, "retrieve", mid, score=sim)
            hits.append(RecallHit(self.store.get(mid), round(sim, 4), from_archive))
        return hits

    def mark_used(self, session_id: str, hits: list[RecallHit], answer: str, query: str = "") -> list[dict]:
        turn, out = self.store.current_turn(session_id), []
        for h in hits:
            ov = answer_overlap(h.memory.text, answer, query)
            used = ov >= self.cfg.used_threshold
            if used:
                self.store.mark_used(h.memory.id, turn)
            self.store.log(session_id, turn, "used" if used else "ignored", h.memory.id, score=ov)
            out.append({"memory_id": h.memory.id, "overlap": round(ov, 4), "used": used})
        return out

    def maintain(self, session_id: str) -> list[dict]:
        """Evict lowest-scoring active memories until back under budget."""
        cfg, turn = self.cfg, self.store.current_turn(session_id)
        active = self.store.list_memories(session_id, ACTIVE)
        over = len(active) - cfg.active_budget
        if over <= 0:
            return []
        # recently touched memories are protected; if everything is protected we allow a temporary overflow
        evictable = [m for m in active if turn - m.last_access_turn >= cfg.grace_turns]
        scored = sorted(((retention_score(m, turn, cfg), m) for m in evictable), key=lambda x: x[0]["score"])
        evicted = []
        for breakdown, m in scored[:over]:
            self._move(session_id, m.id, ARCHIVE)
            self.store.log(session_id, turn, "evict", m.id, score=breakdown["score"], detail=breakdown)
            evicted.append({"memory_id": m.id, "text": m.text, **breakdown})
        return evicted

    def snapshot(self, session_id: str) -> list[dict]:
        turn = self.store.current_turn(session_id)
        rows = [
            {**m.__dict__, **retention_score(m, turn, self.cfg)} for m in self.store.list_memories(session_id)
        ]
        return sorted(rows, key=lambda r: (r["tier"] != ACTIVE, -r["score"]))

    def _move(self, session_id: str, memory_id: str, tier: str, turn: int | None = None):
        self.store.set_tier(memory_id, tier, turn)
        self.index.set_tier(memory_id, session_id, tier)
