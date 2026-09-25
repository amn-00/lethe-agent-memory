"""Grading. Two scorers:

- regex: deterministic and free, but can't grade ordering ("before Farah it was Omkar"
  contains "farah" and still passes).
- judge: an LLM compares the answer with a reference answer. Used as the primary score
  when available; the two are reported side by side with their agreement rate.
"""

import re

_DASHES = re.compile(r"[\u2010-\u2015\u2212\u2011]")
_SPACES = re.compile(r"[\u00a0\u202f\u2009\u2007]")


def normalize(text: str) -> str:
    """Models often emit non-breaking hyphens (P2\u2011117) and thin spaces (VS\u202fCode)."""
    return _SPACES.sub(" ", _DASHES.sub("-", text)).replace("**", "")


def regex_correct(answer: str, patterns: list[str]) -> bool:
    text = normalize(answer)
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


JUDGE_PROMPT = """You are grading an assistant's answer to a question about facts the user shared earlier.

Question: {question}
Reference answer: {gold}
Assistant's answer: {answer}

The answer is CORRECT only if it gives the reference answer as its answer to the question.
Mentioning the reference while giving a different or reversed answer is INCORRECT.
Extra detail or different wording is fine. Declining to answer is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def parse_verdict(text: str) -> bool | None:
    t = (text or "").strip().upper()
    if "INCORRECT" in t:
        return False
    if "CORRECT" in t:
        return True
    return None  # unparseable; caller falls back to regex


async def judge_correct(llm, question: str, gold: str, answer: str) -> bool | None:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=normalize(answer))
    return parse_verdict(await llm.chat([{"role": "user", "content": prompt}]))
