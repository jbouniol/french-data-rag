"""Strict judgment parsing and HTTP clients, using Python's standard library."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SCORE_NAMES = ("correctness", "groundedness", "citation_support")
FLAG_NAMES = ("abstention_ok", "injection_resisted")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


# An API credential is scoped to the selected endpoint, not redirect targets.
# The default HTTPS handler retains certificate and hostname verification.
urlopen = build_opener(_NoRedirect()).open


def validate_verdict(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Le jugement doit être un objet JSON.")
    scores = value.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(SCORE_NAMES + FLAG_NAMES):
        raise ValueError("Le jugement doit contenir exactement les cinq champs scores documentés.")
    for name in SCORE_NAMES:
        score = scores[name]
        if score is None and name != "correctness":
            continue
        if type(score) is not int or score not in (0, 1, 2):
            raise ValueError(f"Score {name} invalide : entier 0, 1 ou 2 requis.")
    for name in FLAG_NAMES:
        if scores[name] is not None and type(scores[name]) is not bool:
            raise ValueError(f"Score {name} invalide : booléen ou null requis.")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
        raise ValueError("reason doit être une justification non vide de 4 000 caractères maximum.")
    return {"scores": scores, "reason": reason}


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    parsed = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("Le JSON du juge contient une clé dupliquée.")
        parsed[key] = value
    return parsed


def validate_base_url(base_url: str, provider: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("L'URL du juge doit utiliser HTTP(S) et avoir un hôte.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("L'URL du juge ne doit contenir ni identifiants, ni paramètres, ni fragment.")
    local = parsed.hostname.lower() == "localhost"
    try:
        local = local or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme == "http" and not local:
        raise ValueError("HTTPS est requis pour un juge distant ; HTTP est réservé à la boucle locale.")
    if provider not in ("ollama", "openai"):
        raise ValueError("Fournisseur non pris en charge.")
    return base_url.rstrip("/")


def make_payload(golden: dict, result: dict, corpus: dict, annotation: dict | None = None) -> dict:
    ids = list(dict.fromkeys(golden["gold_docs"] + result["citations"] + result.get("context_doc_ids", [])))
    missing = [doc_id for doc_id in golden["gold_docs"] + result.get("context_doc_ids", []) if doc_id not in corpus]
    if missing:
        raise ValueError("Identifiants absents du corpus : " + ", ".join(missing))
    evidence = []
    for doc_id in ids:
        document = corpus.get(doc_id, {})
        evidence.append({
            "doc_id": doc_id,
            "resolvable": doc_id in corpus,
            "roles": [name for name, group in (
                ("reference", golden["gold_docs"]), ("cited", result["citations"]),
                ("context", result.get("context_doc_ids", []))) if doc_id in group],
            **{name: document[name] for name in ("text", "article_number", "valid_from", "valid_to") if name in document},
        })
    annotation = annotation or {}
    return {
        "id": golden["id"], "question": golden["question"],
        "reference_answer": golden["answer"], "gold_docs": golden["gold_docs"],
        "candidate": {name: result[name] for name in ("answer", "citations", "calculation", "stop_reason") if name in result},
        "annotation": {name: annotation[name] for name in ("category", "as_of", "expected_calculation", "expected_behavior") if name in annotation},
        "evidence": evidence,
    }


def judge_request(*, provider: str, base_url: str, model: str, prompt: str,
                  payload: dict, api_key: str | None = None, timeout: float = 120,
                  max_tokens: int = 2048, seed: int = 42) -> tuple[dict, dict]:
    base_url = validate_base_url(base_url, provider)
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    if provider == "ollama":
        endpoint = base_url + "/api/chat"
        body = {"model": model, "messages": messages, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens}}
    else:
        endpoint = base_url + "/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0,
                "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
    except HTTPError as exc:
        # Do not log server response bodies: they may echo request credentials.
        raise RuntimeError(f"Le juge a répondu HTTP {exc.code}; vérifier service, modèle et accès.") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Le service du juge est inaccessible ou le délai est dépassé.") from None
    try:
        if provider == "ollama":
            content = raw["message"]["content"]
            usage = {"tokens_in": raw.get("prompt_eval_count"), "tokens_out": raw.get("eval_count")}
            complete = raw.get("done") is True and raw.get("done_reason") != "length"
        else:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            usage = {"tokens_in": raw.get("usage", {}).get("prompt_tokens"),
                     "tokens_out": raw.get("usage", {}).get("completion_tokens")}
            complete = choice.get("finish_reason") == "stop"
        if not complete:
            raise ValueError("Le juge n'a pas produit une réponse complète.")
        verdict = validate_verdict(json.loads(content, object_pairs_hook=_unique_keys))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise ValueError("Réponse du service ou JSON du jugement invalide.") from None
    metadata = {
        "provider": provider, "model_requested": model, "model_returned": raw.get("model"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "temperature": 0, "seed": seed if provider == "ollama" else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "usage": usage,
    }
    return verdict, metadata
