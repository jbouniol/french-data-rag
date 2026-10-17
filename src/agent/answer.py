"""Agent interface: answer(question) -> AnswerTrace."""
from dataclasses import asdict, dataclass, field
import json
import sys


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


def answer(question: str, mode: str = "rag") -> AnswerTrace:
    """mode: rag (search -> LLM, a fixed workflow) | agent (tool calls in a loop)"""
    if mode == "rag":
        from src.agent import rag
        return AnswerTrace(**rag.run(question))
    if mode == "agent":
        from src.agent import agent
        return AnswerTrace(**agent.run(question))
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    # uv run python -m src.agent.answer [--agent] "question"
    args = sys.argv[1:]
    mode = "agent" if "--agent" in args else "rag"
    question = " ".join(a for a in args if a != "--agent") or "Quelle est la durée maximale d'un contrat d'apprentissage ?"
    print(json.dumps(asdict(answer(question, mode=mode)), ensure_ascii=False, indent=2))
