"""Local LLM backend: one Ollama chat call -> (message, usage)."""
from dataclasses import dataclass
from functools import lru_cache
import os
import time

import ollama

MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b-it-qat")
NUM_CTX = 8192          # Ollama's default (4096) is too small for 5-8 articles + instructions


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    def __add__(self, other):
        return Usage(self.tokens_in + other.tokens_in,
                     self.tokens_out + other.tokens_out,
                     self.latency_ms + other.latency_ms)


@lru_cache(maxsize=1)
def get_client():
    return ollama.Client()   # reads OLLAMA_HOST, defaults to http://localhost:11434


def chat(messages, tools=None, schema=None, max_tokens=1024):
    """One deterministic chat call. `schema` = JSON schema for constrained decoding."""
    start = time.perf_counter()
    response = get_client().chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        format=schema,
        think=False,
        keep_alive="30m",   # loading the model takes minutes on a 16 GB laptop: keep it during an eval
        options={"temperature": 0, "seed": 0, "num_ctx": NUM_CTX, "num_predict": max_tokens},
    )
    usage = Usage(response.prompt_eval_count or 0, response.eval_count or 0,
                  (time.perf_counter() - start) * 1000)
    return response.message, usage
