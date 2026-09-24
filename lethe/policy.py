import math
import re

from .models import MemoryRecord, PolicyConfig


def retention_score(m: MemoryRecord, turn: int, cfg: PolicyConfig) -> dict:
    """Higher = keep. Returns the full breakdown so every eviction is explainable."""
    age = max(0, turn - m.last_access_turn)
    recency = 0.5 ** (age / cfg.half_life_turns)
        # counts USES, not retrievals: being pulled in and ignored must never raise a score
    frequency = 1 - 1 / (1 + m.used_count)                
    used_rate = (m.used_count + 1) / (m.retrieved_count + 2)   # Laplace-smoothed, 0.5 prior
    score = cfg.w_recency * recency + cfg.w_frequency * frequency + cfg.w_used * used_rate
    return {
        "score": round(score, 4),
        "recency": round(recency, 4),
        "frequency": round(frequency, 4),
        "used_rate": round(used_rate, 4),
    }


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this", "that", "was", "have",
    "has", "had", "from", "they", "she", "his", "her", "its", "our", "will", "would", "can", "could",
    "what", "when", "where", "which", "who", "how", "about", "into", "than", "then", "there", "their",
    "been", "also", "just", "like", "some", "any", "all", "one",
}


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def answer_overlap(memory_text: str, answer: str, query: str = "") -> float:
    """Share of a memory's content words that show up in the answer.
    Words already in the query are excluded: an answer echoing the question
    is not evidence that a memory was used.
    Cheap and deterministic; misses paraphrases, which is a known limit."""
    m = tokens(memory_text) - tokens(query)
    if not m:
        return 0.0
    return len(m & tokens(answer)) / len(m)
