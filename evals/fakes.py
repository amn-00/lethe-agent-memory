"""Offline stand-ins so the eval pipeline can be tested without API calls or model downloads."""

import hashlib
import math

from lethe.policy import tokens


class FakeEmbedder:
    """Hashed bag-of-words; texts sharing words get high cosine similarity."""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in tokens(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class MemoryEchoLLM:
    """'Answers' by echoing everything in its context except the question.
    So it's right exactly when the needed fact made it into the prompt, which is
    what the eval is measuring for memory systems."""

    last_usage = None

    async def chat(self, messages):
        return " | ".join(m["content"] for m in messages[:-1])
