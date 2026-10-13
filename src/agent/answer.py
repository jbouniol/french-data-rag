"""Agent interface (stub, implemented by the agent workstream)."""
from dataclasses import dataclass, field


@dataclass
class AnswerTrace:
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    cost_eur: float = 0.0


def answer(question: str) -> AnswerTrace:
    """Answer a question using the retrieval."""
    raise NotImplementedError
