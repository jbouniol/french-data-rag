"""OpenRouter Cloud API bridge for Workstream E (Local Ollama vs Cloud API comparison).

Dynamically loads `OPENROUTER_API_KEY` from `.env` (hot-reload on every request, no server
restart needed) and bridges `src.agent.llm.chat` to `https://openrouter.ai/api/v1/chat/completions`
so that `src/agent/agent.py` and `src/agent/rag.py` run identically on local Ollama models
or OpenRouter models (OpenAI, Anthropic, Google Gemini, Mistral, Llama) without modifying
`src/agent/`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

from src.agent import llm

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
USD_TO_EUR = 0.92

# Curated Cloud API models (OpenRouter & Anthropic Direct) for Workstream E
API_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    # OpenRouter API models (OPENROUTER_API_KEY)
    "google/gemini-2.5-flash": {
        "provider": "openrouter",
        "label": "gemini-2.5-flash",
        "in_eur_per_1m": 0.14,
        "out_eur_per_1m": 0.56,
    },
    "openai/gpt-4o-mini": {
        "provider": "openrouter",
        "label": "gpt-4o-mini",
        "in_eur_per_1m": 0.14,
        "out_eur_per_1m": 0.56,
    },
    "mistralai/mistral-small-3.2-24b-instruct": {
        "provider": "openrouter",
        "label": "mistral-small-3.2-24b",
        "in_eur_per_1m": 0.10,
        "out_eur_per_1m": 0.30,
    },
    "openai/gpt-4o": {
        "provider": "openrouter",
        "label": "gpt-4o",
        "in_eur_per_1m": 2.30,
        "out_eur_per_1m": 9.20,
    },
    # Anthropic Direct API models (ANTHROPIC_API_KEY)
    "claude-sonnet-5-5": {
        "provider": "anthropic",
        "label": "claude-sonnet-5-5",
        "in_eur_per_1m": 2.76,
        "out_eur_per_1m": 13.80,
    },
    "claude-haiku-5-5": {
        "provider": "anthropic",
        "label": "claude-haiku-5-5",
        "in_eur_per_1m": 0.92,
        "out_eur_per_1m": 4.60,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "label": "claude-sonnet-4-6",
        "in_eur_per_1m": 2.76,
        "out_eur_per_1m": 13.80,
    },
}

_MODEL_ALIASES: dict[str, str] = {
    "anthropic/claude-3.5-haiku": "claude-haiku-5-5",
    "anthropic/claude-3.7-sonnet": "claude-sonnet-4-6",
    "anthropic/claude-haiku-4.5": "claude-haiku-5-5",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-6",
    "claude-haiku-4-5": "claude-haiku-5-5",
    "claude-sonnet-4-5": "claude-sonnet-4-6",
    "claude-3-5-haiku-latest": "claude-haiku-5-5",
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
}

_ORIGINAL_OLLAMA_CHAT = llm.chat
_LAST_RUN_COST_EUR: float = 0.0


@dataclass
class CompatFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class CompatToolCall:
    function: CompatFunction
    id: str = "call_1"


class CompatMessage(dict):
    """Message object compatible with both Ollama's `response.message` and dict access."""

    def __init__(
        self,
        role: str = "assistant",
        content: str = "",
        tool_calls: list[CompatToolCall] | None = None,
        openai_tool_calls: list[dict[str, Any]] | None = None,
        anthropic_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self._openai_tool_calls = openai_tool_calls
        self._anthropic_blocks = anthropic_blocks


def read_dotenv_vars() -> dict[str, str]:
    """Read key-value pairs directly from `.env` on disk (hot-reload on every check)."""
    env_vars: dict[str, str] = {}
    if ENV_PATH.exists():
        try:
            for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    env_vars[k] = v
        except Exception:
            pass
    return env_vars


def get_openrouter_config() -> tuple[str, str]:
    """Return `(api_key, base_url)` from `.env` or environment."""
    dotenv = read_dotenv_vars()

    def _get(name: str) -> str:
        return (dotenv.get(name) or os.environ.get(name) or "").strip()

    api_key = _get("OPENROUTER_API_KEY") or _get("API_KEY") or _get("OPENAI_API_KEY")
    base_url = (_get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    return api_key, base_url


def get_anthropic_key() -> str:
    """Return `ANTHROPIC_API_KEY` from `.env` or environment."""
    dotenv = read_dotenv_vars()
    return (dotenv.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def get_configured_api_keys() -> dict[str, str]:
    """Return active provider keys (`openrouter`, `anthropic`) configured in `.env`."""
    keys: dict[str, str] = {}
    or_key, _ = get_openrouter_config()
    if or_key:
        keys["openrouter"] = or_key
    ant_key = get_anthropic_key()
    if ant_key:
        keys["anthropic"] = ant_key
    return keys


def get_model_provider(model_id: str) -> str:
    """Return `'anthropic'`, `'openrouter'`, or `'local'` for a given `model_id`."""
    resolved = _MODEL_ALIASES.get(str(model_id or ""), str(model_id or ""))
    meta = API_MODEL_CATALOG.get(resolved)
    if meta:
        return str(meta.get("provider") or "openrouter")
    if resolved.startswith("claude-"):
        return "anthropic"
    if "/" in resolved:
        return "openrouter"
    return "local"


def get_available_api_models() -> list[str]:
    """Return the list of cloud model IDs whose provider key is configured."""
    keys = get_configured_api_keys()
    return [
        m_id
        for m_id, meta in API_MODEL_CATALOG.items()
        if meta.get("provider") in keys
    ]


def is_api_model(model_id: str) -> bool:
    """True if `model_id` is an OpenRouter or Anthropic cloud model."""
    return get_model_provider(model_id) in ("openrouter", "anthropic")


def compute_api_cost_eur(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Compute the EUR cost of a run for cloud API models (0.0 for local Ollama)."""
    global _LAST_RUN_COST_EUR
    if not is_api_model(model_id):
        return 0.0
    if _LAST_RUN_COST_EUR > 0:
        cost = _LAST_RUN_COST_EUR
        _LAST_RUN_COST_EUR = 0.0
        return round(cost, 6)
    resolved = _MODEL_ALIASES.get(model_id, model_id)
    meta = API_MODEL_CATALOG.get(resolved, {"in_eur_per_1m": 0.25, "out_eur_per_1m": 1.00})
    cost = (tokens_in / 1_000_000.0) * meta["in_eur_per_1m"] + (tokens_out / 1_000_000.0) * meta["out_eur_per_1m"]
    return round(cost, 6)


def save_api_key_to_dotenv(raw_key: str, provider: str = "openrouter") -> dict[str, Any]:
    """Save `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` into `.env`."""
    key = raw_key.strip()
    prov = (provider or "").strip().lower()
    if prov == "anthropic" or key.startswith("sk-ant-"):
        target_var = "ANTHROPIC_API_KEY"
    else:
        target_var = "OPENROUTER_API_KEY"

    lines: list[str] = []
    replaced = False
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"{target_var}="):
                lines[i] = f"{target_var}={key}"
                replaced = True
                break
    if not replaced:
        lines.append(f"{target_var}={key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "saved_var": target_var,
        "providers": list(get_configured_api_keys().keys()),
        "api_models": get_available_api_models(),
    }


def _strip_markdown_json_fence(text: str) -> str:
    """Strip ```json ... ``` fences if a cloud model wraps JSON output in markdown."""
    cleaned = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return cleaned


def _chat_openrouter(
    model_id: str,
    api_key: str,
    base_url: str,
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
) -> tuple[CompatMessage, llm.Usage]:
    global _LAST_RUN_COST_EUR
    start = time.perf_counter()
    api_messages: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []

    for msg in messages:
        if isinstance(msg, CompatMessage) and msg._openai_tool_calls:
            pending_call_ids = [tc.get("id", f"call_{i+1}") for i, tc in enumerate(msg._openai_tool_calls)]
            api_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg._openai_tool_calls,
            })
            continue

        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""

        if role in ("system", "user", "assistant"):
            api_messages.append({"role": role, "content": str(content)})
        elif role == "tool":
            call_id = pending_call_ids.pop(0) if pending_call_ids else "call_1"
            tool_name = msg.get("tool_name") if isinstance(msg, dict) else getattr(msg, "tool_name", "tool")
            api_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": str(tool_name),
                "content": str(content),
            })

    if schema is not None and api_messages:
        # Ensure the model knows the exact JSON schema keys expected
        props = list((schema.get("properties") or {}).keys())
        schema_hint = (
            f"\n\nRespond ONLY with a valid JSON object matching these exact keys: {json.dumps(props)}. "
            "Do not wrap in markdown code fences."
        )
        if api_messages[-1]["role"] == "user":
            api_messages[-1] = {
                "role": "user",
                "content": api_messages[-1]["content"] + schema_hint,
            }

    resolved_model_id = _MODEL_ALIASES.get(model_id, model_id)
    payload: dict[str, Any] = {
        "model": resolved_model_id,
        "messages": api_messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "required"
    if schema is not None and not resolved_model_id.startswith(("anthropic/", "meta-llama/")):
        payload["response_format"] = {"type": "json_object"}

    endpoint_url = f"{base_url}/chat/completions"
    req = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "Alter-Ego Code du Travail RAG",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {err.code}: {err_body[:260]}") from err

    choices = data.get("choices") or []
    message_data = (choices[0].get("message") or {}) if choices else {}
    content_text = str(message_data.get("content") or "")
    if schema is not None and content_text:
        content_text = _strip_markdown_json_fence(content_text)

    raw_tool_calls = message_data.get("tool_calls") or []
    parsed_calls: list[CompatToolCall] = []
    for idx, tc in enumerate(raw_tool_calls):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                args_dict = {}
        else:
            args_dict = dict(raw_args)
        parsed_calls.append(
            CompatToolCall(
                function=CompatFunction(name=str(fn.get("name") or ""), arguments=args_dict),
                id=str(tc.get("id") or f"call_{idx + 1}"),
            )
        )

    usage_data = data.get("usage") or {}
    tokens_in = int(usage_data.get("prompt_tokens") or 0)
    tokens_out = int(usage_data.get("completion_tokens") or 0)
    # OpenRouter includes `cost` (in USD) inside `usage` when available
    if usage_data.get("cost") is not None:
        try:
            _LAST_RUN_COST_EUR += float(usage_data["cost"]) * USD_TO_EUR
        except (TypeError, ValueError):
            pass

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    msg_obj = CompatMessage(
        role="assistant",
        content=content_text,
        tool_calls=parsed_calls,
        openai_tool_calls=raw_tool_calls if raw_tool_calls else None,
    )
    return msg_obj, llm.Usage(tokens_in, tokens_out, elapsed_ms)


def _chat_anthropic_direct(
    model_id: str,
    api_key: str,
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
) -> tuple[CompatMessage, llm.Usage]:
    """Direct Anthropic Messages API (`https://api.anthropic.com/v1/messages`) using `ANTHROPIC_API_KEY`."""
    start = time.perf_counter()
    resolved_model = _MODEL_ALIASES.get(model_id, model_id)
    if resolved_model.startswith("anthropic/"):
        resolved_model = resolved_model.split("/", 1)[1].replace(".", "-")

    system_parts: list[str] = []
    ant_messages: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []

    for msg in messages:
        if isinstance(msg, CompatMessage) and msg._anthropic_blocks:
            pending_tool_ids = [
                b.get("id", f"toolu_{i+1}")
                for i, b in enumerate(msg._anthropic_blocks)
                if b.get("type") == "tool_use"
            ]
            ant_messages.append({"role": "assistant", "content": msg._anthropic_blocks})
            continue

        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""

        if role == "system":
            system_parts.append(str(content))
        elif role == "user":
            ant_messages.append({"role": "user", "content": str(content)})
        elif role == "assistant":
            ant_messages.append({"role": "assistant", "content": str(content) or "..."})
        elif role == "tool":
            tool_id = pending_tool_ids.pop(0) if pending_tool_ids else "toolu_1"
            block = {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(content),
            }
            if ant_messages and ant_messages[-1]["role"] == "user" and isinstance(ant_messages[-1]["content"], list):
                ant_messages[-1]["content"].append(block)
            else:
                ant_messages.append({"role": "user", "content": [block]})

    if schema is not None and ant_messages:
        props = list((schema.get("properties") or {}).keys())
        schema_hint = (
            f"\n\nRespond ONLY with a valid JSON object matching these exact keys: {json.dumps(props)}. "
            "Do not wrap in markdown code fences."
        )
        if ant_messages[-1]["role"] == "user" and isinstance(ant_messages[-1]["content"], str):
            ant_messages[-1]["content"] = ant_messages[-1]["content"] + schema_hint
        else:
            ant_messages.append({"role": "user", "content": schema_hint.strip()})

    payload: dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": max_tokens,
        "messages": ant_messages,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if tools:
        ant_tools: list[dict[str, Any]] = []
        for t in tools:
            fn = t.get("function") or {}
            ant_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        payload["tools"] = ant_tools

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {err.code}: {err_body[:260]}") from err

    blocks = data.get("content") or []
    text_parts: list[str] = []
    parsed_calls: list[CompatToolCall] = []

    for idx, b in enumerate(blocks):
        b_type = b.get("type")
        if b_type == "text":
            text_parts.append(str(b.get("text") or ""))
        elif b_type == "tool_use":
            parsed_calls.append(
                CompatToolCall(
                    function=CompatFunction(
                        name=str(b.get("name") or ""),
                        arguments=dict(b.get("input") or {}),
                    ),
                    id=str(b.get("id") or f"toolu_{idx + 1}"),
                )
            )

    content_text = "\n".join(text_parts).strip()
    if schema is not None and content_text:
        content_text = _strip_markdown_json_fence(content_text)

    usage_data = data.get("usage") or {}
    tokens_in = int(usage_data.get("input_tokens") or 0)
    tokens_out = int(usage_data.get("output_tokens") or 0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    msg_obj = CompatMessage(
        role="assistant",
        content=content_text,
        tool_calls=parsed_calls,
        anthropic_blocks=blocks if parsed_calls else None,
    )
    return msg_obj, llm.Usage(tokens_in, tokens_out, elapsed_ms)


import threading
from typing import Callable

_EVENT_LOCAL = threading.local()


def set_live_event_callback(cb: Callable[[dict[str, Any]], None] | None) -> None:
    """Register a per-thread callback receiving real-time execution events."""
    _EVENT_LOCAL.callback = cb


def emit_live_event(event: dict[str, Any]) -> None:
    """Emit a real-time event if a callback is active on the current thread."""
    cb = getattr(_EVENT_LOCAL, "callback", None)
    if callable(cb):
        try:
            cb(event)
        except Exception:
            pass


def _describe_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "search_articles":
        q = args.get("query") or ""
        return f"Recherche hybride (BM25 + bge-m3 + reranker) : « {q} »"
    if name == "compute_remuneration":
        age = args.get("age", "?")
        yr = args.get("contract_year", "?")
        lic = ", licence pro" if args.get("licence_pro") else ""
        return f"Calcul déterministe grille SMIC D6222-26 ({age} ans, année {yr}{lic})"
    if name == "get_article_at_date":
        num = args.get("num", "?")
        dt_str = args.get("date", "?")
        return f"Vérification historique Légifrance : Art. {num} au {dt_str}"
    if name == "final_answer":
        return "Validation finale & contrôle des citations (check_final)"
    return f"Exécution de {name}"


def _sanitize_final_answer_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common small-model citation quirks (e.g. [compute_remuneration] -> [D6222-26])."""
    if not isinstance(args, dict):
        return args
    ans = args.get("answer")
    if isinstance(ans, str) and ans:
        # Replace [compute_remuneration] with [D6222-26]
        ans = re.sub(r"\[\s*compute_remuneration\s*\]", "[D6222-26]", ans, flags=re.IGNORECASE)
        # Replace [Article D6222-26] or [Art. L6222-35] with [D6222-26] / [L6222-35]
        ans = re.sub(r"\[\s*(?:article|art\.?)\s+([LRD]\.?\s*\d[\w-]*)\s*\]", r"[\1]", ans, flags=re.IGNORECASE)
        args["answer"] = ans

    cites = args.get("citations")
    if isinstance(cites, list):
        cleaned_cites: list[Any] = []
        for c in cites:
            if not isinstance(c, str):
                cleaned_cites.append(c)
                continue
            c_str = c.strip()
            if c_str.lower() == "compute_remuneration":
                c_str = "D6222-26"
            elif c_str.lower() in ("search_articles", "get_article_at_date", "final_answer"):
                continue
            else:
                c_str = re.sub(r"^(?:article|art\.?)\s+", "", c_str, flags=re.IGNORECASE).strip()
            if c_str and c_str not in cleaned_cites:
                cleaned_cites.append(c_str)
        args["citations"] = cleaned_cites
    return args


def unified_chat(messages: Any, tools: Any = None, schema: Any = None, max_tokens: int = 1024) -> tuple[Any, llm.Usage]:
    """Drop-in replacement for `src.agent.llm.chat` supporting local Ollama, OpenRouter, and Anthropic Direct."""
    model_id = llm.MODEL
    short_model = model_id.split("/")[-1] if "/" in model_id else model_id

    # Accurate live phase emission before the LLM call begins
    msg_list = list(messages) if isinstance(messages, (list, tuple)) else []
    tool_msgs = [
        m for m in msg_list
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "tool"
    ]

    if schema is not None:
        emit_live_event({
            "type": "phase",
            "phase": "generating",
            "title": f"Rédaction structurée ({short_model})",
            "detail": "Génération JSON sous contrainte de schéma & citations Légifrance",
        })
    elif tools is not None and len(tool_msgs) == 0:
        emit_live_event({
            "type": "phase",
            "phase": "thinking",
            "title": f"Réflexion & planification ({short_model})",
            "detail": "Analyse de la question et sélection des outils juridiques",
        })
    elif tools is not None:
        emit_live_event({
            "type": "phase",
            "phase": "reasoning",
            "title": f"Analyse des sources & synthèse ({short_model})",
            "detail": f"Évaluation des {len(tool_msgs)} résultat(s) d'outils et vérification des articles",
        })

    t0 = time.perf_counter()
    provider = get_model_provider(model_id)
    if provider == "local":
        msg, usage = _ORIGINAL_OLLAMA_CHAT(messages, tools=tools, schema=schema, max_tokens=max_tokens)
    elif provider == "anthropic":
        ant_key = get_anthropic_key()
        if not ant_key:
            raise RuntimeError(
                "Clé ANTHROPIC_API_KEY manquante dans le fichier .env. "
                "Cliquez sur le badge en haut à droite ou ajoutez ANTHROPIC_API_KEY=sk-ant-... dans .env."
            )
        msg, usage = _chat_anthropic_direct(
            model_id=model_id,
            api_key=ant_key,
            messages=messages,
            tools=tools,
            schema=schema,
            max_tokens=max_tokens,
        )
    else:
        api_key, base_url = get_openrouter_config()
        if not api_key:
            raise RuntimeError(
                "Clé OPENROUTER_API_KEY manquante dans le fichier .env. "
                "Cliquez sur le badge en haut à droite ou ajoutez OPENROUTER_API_KEY=sk-or-v1-... dans .env."
            )
        msg, usage = _chat_openrouter(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            tools=tools,
            schema=schema,
            max_tokens=max_tokens,
        )

    # Sanitize schema-constrained JSON output if it's a FinalAnswer
    if schema is not None and getattr(msg, "content", None):
        try:
            parsed_json = json.loads(msg.content)
            if isinstance(parsed_json, dict) and "answer" in parsed_json:
                _sanitize_final_answer_args(parsed_json)
                new_content = json.dumps(parsed_json, ensure_ascii=False)
                msg.content = new_content
                if isinstance(msg, dict):
                    msg["content"] = new_content
        except Exception:
            pass

    llm_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    raw_tcs = getattr(msg, "tool_calls", None) or []
    planned = []
    for tc in raw_tcs:
        fn = getattr(tc, "function", None)
        if fn:
            fn_name = getattr(fn, "name", "")
            fn_args = dict(getattr(fn, "arguments", {}) or {})
            if fn_name == "final_answer":
                fn_args = _sanitize_final_answer_args(fn_args)
                try:
                    fn.arguments = fn_args
                except Exception:
                    pass
            planned.append({
                "name": fn_name,
                "arguments": fn_args,
                "description": _describe_tool_call(fn_name, fn_args),
            })

    if planned:
        emit_live_event({
            "type": "llm_decision",
            "latency_ms": llm_ms,
            "tokens_in": usage.tokens_in,
            "tokens_out": usage.tokens_out,
            "planned_tools": planned,
        })

    return msg, usage


def install_llm_bridge() -> None:
    """Install `unified_chat` onto `src.agent.llm.chat` and wrap tool/RAG execution for live telemetry."""
    llm.chat = unified_chat

    from src.agent import agent as agent_mod
    from src.agent import rag as rag_mod

    if not getattr(agent_mod, "_live_wrapped", False):
        orig_run_tool = agent_mod.run_tool

        def wrapped_run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            emit_live_event({
                "type": "tool_start",
                "name": name,
                "arguments": args,
                "title": f"Exécution de {name}",
                "detail": _describe_tool_call(name, args),
            })
            t0 = time.perf_counter()
            out = orig_run_tool(name, args)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            # Wrap compute_remuneration output in <source article="D6222-26"> so small models cite [D6222-26]
            if name == "compute_remuneration" and not out.get("error") and out.get("content"):
                if "<source" not in str(out["content"]):
                    out = dict(out)
                    out["content"] = f'<source article="D6222-26">\n{out["content"]}\n</source>'
            citable_nums = list((out.get("citable") or {}).keys())
            emit_live_event({
                "type": "tool_done",
                "name": name,
                "arguments": args,
                "latency_ms": elapsed_ms,
                "citable_nums": citable_nums,
                "error": bool(out.get("error")),
                "result": str(out.get("content") or "")[:300],
            })
            return out

        agent_mod.run_tool = wrapped_run_tool
        agent_mod._live_wrapped = True  # type: ignore[attr-defined]

    if not getattr(rag_mod, "_live_wrapped", False):
        orig_rag_search = rag_mod.search

        def wrapped_rag_search(question: str, k: int = 5, filters: Any = None) -> Any:
            emit_live_event({
                "type": "tool_start",
                "name": "search_articles",
                "arguments": {"query": question, "k": k},
                "title": "Recherche hybride (BM25 + bge-m3 + reranker)",
                "detail": f"Recherche des {k} articles VIGUEUR les plus pertinents",
            })
            t0 = time.perf_counter()
            hits = orig_rag_search(question, k=k, filters=filters) if filters is not None else orig_rag_search(question, k=k)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            nums = [h.metadata.get("num", "") for h in hits if hasattr(h, "metadata")]
            emit_live_event({
                "type": "tool_done",
                "name": "search_articles",
                "arguments": {"query": question, "k": k},
                "latency_ms": elapsed_ms,
                "citable_nums": nums,
                "error": False,
                "result": f"Articles trouvés : {', '.join(nums)}",
            })
            return hits

        rag_mod.search = wrapped_rag_search
        rag_mod._live_wrapped = True  # type: ignore[attr-defined]


