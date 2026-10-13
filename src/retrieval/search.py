"""Search interface (stub, implemented by the retrieval workstream)."""
from dataclasses import dataclass, field


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


def search(query: str, k: int = 10, filters: dict | None = None) -> list[Hit]:
    """Return the k best chunks for the query, with optional filters."""
    raise NotImplementedError
