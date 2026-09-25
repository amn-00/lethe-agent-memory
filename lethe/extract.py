"""Fact extraction: store durable facts instead of raw messages.

Raw messages cause two problems the eval exposed:
  - cold start: at save time "i work at zoho" and "lol ok" look identical, so facts got evicted first
  - phrasing mismatch: "started a new book, the three body problem" never matched "which book am i reading?"

Extraction rewrites facts as standalone first-person sentences and drops chatter entirely.
Messages are batched (one LLM call per `batch_size` messages, or right before an answer is
needed) so the cost stays around one extra call per several messages.
"""

import json
import re

from .agent import estimate_tokens

EXTRACT_PROMPT = """You extract durable personal facts from a user's chat messages, for a long-term memory.

{known}Messages, each tagged with its turn:
{messages}

Rules:
- Keep only facts about the user or their life that could matter later: people, places, jobs, possessions, preferences, plans, dates, numbers, codes, and changes to any of these.
- Skip small talk, moods, passing remarks about the moment (weather, snacks, tiredness), and questions.
- Write each fact as a short first-person sentence that stands on its own, e.g. "My sister lives in Pune."
- Keep names, numbers and codes exactly as given.
- If a message changes an earlier fact (including one of the already-known facts), state the new situation and what it replaced, using the same wording as the question someone would ask, e.g. "I now live in Hyderabad (moved from Indore)." or "I'm now reading Dune (finished Neuromancer)."
- Don't repeat already-known facts that haven't changed.
- Tag each fact with the turn of the message it came from.

Return only JSON, no other text: {{"facts": [{{"turn": 3, "fact": "My sister lives in Pune."}}]}}
If there are no facts, return {{"facts": []}}"""


def parse_facts(text: str, default_turn: int) -> list[tuple[int, str]] | None:
    """Lenient JSON parsing (tolerates code fences / stray text). None means unparseable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    items = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        try:
            turn = int(item.get("turn", default_turn))
        except (TypeError, ValueError):
            turn = default_turn
        if fact:
            out.append((turn, fact))
    return out


class FactExtractor:
    def __init__(self, llm, batch_size: int = 8):
        self.llm = llm
        self.batch_size = batch_size
        self.calls = 0
        self.tokens = 0
        self.failures = 0

    async def extract(self, batch: list[tuple[int, str]], known: list[str] | None = None) -> list[tuple[int, str]]:
        """batch: [(turn, message)] -> [(turn, fact)]. `known`: related facts already in memory, so updates
        that span batches ("finished dune" in one batch, "started a new book" in the next) are recognised.
        On unparseable output, falls back to the raw messages so nothing is silently lost."""
        lines = "\n".join(f"[t{turn}] {text}" for turn, text in batch)
        known_block = ""
        if known:
            known_block = "Facts already in memory (context only):\n" + "\n".join(f"- {k}" for k in known) + "\n\n"
        prompt = EXTRACT_PROMPT.format(messages=lines, known=known_block)
        messages = [{"role": "user", "content": prompt}]
        reply = await self.llm.chat(messages)
        usage = getattr(self.llm, "last_usage", None) or {}
        self.calls += 1
        self.tokens += usage.get("total_tokens") or estimate_tokens(messages) + len(reply or "") // 4
        facts = parse_facts(reply, default_turn=batch[-1][0])
        if facts is None:
            self.failures += 1
            return list(batch)
        valid_turns = {t for t, _ in batch}
        return [(t if t in valid_turns else batch[-1][0], f) for t, f in facts]
