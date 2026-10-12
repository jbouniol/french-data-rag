#!/usr/bin/env python3
"""Judge an existing RAG run; this script never calls the evaluated assistant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from p4.io import index_by_id, load_corpus, load_golden, load_results, read_jsonl, write_jsonl
from p4.judging import judge_request, make_payload, validate_base_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Ollama: http://127.0.0.1:11434; API compatible: URL finissant par /v1")
    parser.add_argument("--api-key-env", help="Nom de la variable contenant la clé ; jamais la valeur")
    parser.add_argument("--prompt", type=Path, default=Path(__file__).with_name("judge_prompt.md"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-payload-chars", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("Le fichier de sortie existe déjà ; choisir un nouveau chemin.")
        if args.timeout <= 0 or args.max_tokens < 1 or args.max_payload_chars < 1:
            raise ValueError("Timeout, max-tokens et max-payload-chars doivent être positifs.")
        base_url = args.base_url or ("http://127.0.0.1:11434" if args.provider == "ollama" else "https://api.openai.com/v1")
        base_url = validate_base_url(base_url, args.provider)
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not api_key:
            raise ValueError("La variable de clé indiquée est absente ou vide.")
        questions = load_golden(args.questions)
        results = load_results(args.results)
        corpus = load_corpus(args.corpus)
        annotations = index_by_id(read_jsonl(args.annotations), "annotations") if args.annotations else {}
        if set(questions) != set(results):
            raise ValueError("Les identifiants des résultats doivent correspondre exactement aux questions.")
        if set(annotations) - set(questions):
            raise ValueError("Les annotations contiennent des identifiants de questions inconnus.")
        prompt = args.prompt.read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError("Le prompt du juge est vide.")
        # Validate every item before sending the first remote request.
        payloads = []
        for qid, question in questions.items():
            payload = make_payload(question, results[qid], corpus, annotations.get(qid))
            if len(json.dumps(payload, ensure_ascii=False)) > args.max_payload_chars:
                raise ValueError(f"{qid}: contexte trop grand ; augmenter explicitement la limite ou revoir les sources.")
            payloads.append((qid, payload))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        errors = 0
        for qid, payload in payloads:
            started = time.perf_counter()
            abort_service = False
            try:
                verdict, metadata = judge_request(provider=args.provider, base_url=base_url,
                    model=args.model, prompt=prompt, payload=payload, api_key=api_key,
                    timeout=args.timeout, max_tokens=args.max_tokens, seed=args.seed)
                row = {"id": qid, "status": "ok", **verdict, "judge": metadata}
            except (ValueError, RuntimeError) as exc:
                errors += 1
                abort_service = isinstance(exc, RuntimeError)
                row = {"id": qid, "status": "error", "error": str(exc), "judge": {
                    "provider": args.provider, "model_requested": args.model,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
            row["judge"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["judge"]["input_sha256"] = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            rows.append(row)
            print(f"{qid}: {row['status']}", file=sys.stderr)
            if abort_service:
                print("Arrêt des appels : corriger l'accès au service avant de relancer.", file=sys.stderr)
                break
        write_jsonl(args.out, rows)
        print(json.dumps({"judged": len(rows) - errors, "errors": errors,
                          "unrun": len(payloads) - len(rows), "out": str(args.out)}, ensure_ascii=False))
        return 1 if errors else 0
    except (ValueError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
