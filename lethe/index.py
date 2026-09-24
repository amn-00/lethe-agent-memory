from typing import Protocol

import chromadb
from chromadb.config import Settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """ONNX embeddings, no torch. bge-small is ~130MB, fits Render's free tier."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


class VectorIndex:
    def __init__(self, path: str | None = "chroma", collection: str = "memories"):
        settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path, settings=settings) if path else chromadb.EphemeralClient(settings=settings)
        self.col = client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )

    def add(self, memory_id: str, embedding: list[float], session_id: str, tier: str):
        self.col.add(ids=[memory_id], embeddings=[embedding], metadatas=[{"session_id": session_id, "tier": tier}])

    def set_tier(self, memory_id: str, session_id: str, tier: str):
        self.col.update(ids=[memory_id], metadatas=[{"session_id": session_id, "tier": tier}])

    def query(self, embedding: list[float], session_id: str, tier: str, k: int) -> list[tuple[str, float]]:
        res = self.col.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"$and": [{"session_id": session_id}, {"tier": tier}]},
        )
        # cosine distance -> similarity
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]
