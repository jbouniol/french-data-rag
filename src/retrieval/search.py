"""Interface de recherche (stub, implémenté par le chantier retrieval)."""
from dataclasses import dataclass, field


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


def search(query: str, k: int = 10, filters: dict | None = None) -> list[Hit]:
    """Retourne les k meilleurs chunks pour la requête, avec filtres optionnels."""
    raise NotImplementedError
