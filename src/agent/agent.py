"""Tool-using agent: native Ollama tool calls in a loop, until final_answer (course session 6).

Termination: final_answer (answered | not_found | out_of_scope), max_iterations, budget_exceeded,
stuck (same call repeated after a warning), error (LLM unreachable).
"""
import json
import re
import time
from pydantic import Field, ValidationError

from src.agent import llm
from src.agent.rag import GUARDRAILS, NOT_FOUND, SCOPE_INTRO, RagAnswer
from src.agent.tools import TOOLS, normalize_num, run_tool
from src.retrieval.search import NUM_BODY

MAX_ITERATIONS = 6
TOKEN_BUDGET = 40_000     # tokens in + out for one question

SYSTEM = f"""{SCOPE_INTRO}

Use the tools, then finish with final_answer:
- search_articles: find the relevant articles; search again with other words if the first results miss.
- get_article_at_date: when the question is about a past date or an older version of an article.
- compute_remuneration: for an apprentice's minimum pay; never compute the percentages yourself.
- final_answer: your answer, once you have the evidence.

final_answer has three fields:
- "answer": in French, 1 to 4 short sentences, ONLY from the tool results. End every sentence with the
  article number that supports it, in square brackets, like [L6222-7-1].
- "citations": every article number you used, e.g. ["L6222-7-1"]. Cite only articles returned by a tool.
{GUARDRAILS}"""


FINALIZE = ('Write your final answer now, as JSON with the fields "answer", "citations" and "out_of_scope", '
            "following the rules for final_answer.")


class FinalAnswer(RagAnswer):
    answer: str = Field(description="The answer in French, each sentence ending with its article number like [L6222-7-1]")
    citations: list[str] = Field(default_factory=list, description="Article numbers used, e.g. ['L6222-7-1']")
    out_of_scope: bool = Field(default=False, description="True if the question is outside the corpus or not about labour law")


def tool_schema(name: str, description: str, args_model) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": args_model.model_json_schema()}}


TOOL_SCHEMAS = [tool_schema(name, desc, args) for name, (_, args, desc) in TOOLS.items()] + [
    tool_schema("final_answer", "Give the final answer in French with the article numbers cited.", FinalAnswer)]

MARKER = re.compile(rf"\[([LRD]\.?\s*{NUM_BODY})\]", re.IGNORECASE)


def check_final(result: FinalAnswer, citable: dict) -> list[str]:
    """Business rules: an answer must cite, and only articles that a tool returned in this run."""
    if result.status != "answered":
        return []
    problems = []
    cited = {normalize_num(c) for c in result.citations} | {normalize_num(m) for m in MARKER.findall(result.answer)}
    unknown = sorted(cited - citable.keys())
    if unknown:
        problems.append(f"articles {unknown} were not returned by any tool: cite only {sorted(citable)}")
    if not cited:
        problems.append(f'no article cited: end every sentence with its article number like [L6222-7-1], '
                        f'or answer exactly "{NOT_FOUND}"')
    return problems


def run(question: str, max_iterations: int = MAX_ITERATIONS, token_budget: int = TOKEN_BUDGET) -> dict:
    start = time.perf_counter()
    question = question.replace("</question", "&lt;/question")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<question>\n{question}\n</question>"}]
    citable, retrieved, tool_calls, seen = {}, [], [], {}
    usage = llm.Usage()

    def result(stop_reason, answer="", citations=()):
        return {"answer": answer, "citations": list(citations),
                "retrieved_ids": list(dict.fromkeys(retrieved)),
                "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "tool_calls": tool_calls, "stop_reason": stop_reason, "cost_eur": 0.0}

    def finalize(args: dict, record: dict):
        """Validate a final answer: the run's result if it passes, else the error to show the model."""
        try:
            final = FinalAnswer.model_validate(args)
            problems = check_final(final, citable)
        except ValidationError as err:
            problems = [f"invalid arguments: {err.errors()[0]['msg']}"]
        if problems:
            record["error"] = True
            return None, "Tool error: final_answer breaks these rules:\n- " + "\n- ".join(problems)
        record["result"] = final.status
        if final.status != "answered":
            return result(final.status, final.answer.strip()), None
        used = [normalize_num(m) for m in MARKER.findall(final.answer)] + [normalize_num(c) for c in final.citations]
        return result("answered", final.answer.strip(), dict.fromkeys(citable[n] for n in used if n in citable)), None

    for step in range(1, max_iterations + 1):
        try:
            message, u = llm.chat(messages, tools=TOOL_SCHEMAS)
        except Exception as err:            # Ollama down, model missing...: fail loudly, keep the trace
            return result("error", f"LLM error: {err}")
        usage += u
        messages.append(message)

        if not message.tool_calls:
            # Plain text instead of a tool call: an invalid call. Gemma 4 E4B often writes its final answer
            # as text ("final_answer{answer: ...}") that Ollama cannot parse. Recover by constraining the
            # decoding to the FinalAnswer schema: the reply is then valid JSON by construction.
            tool_calls.append({"step": step, "name": None, "arguments": None, "error": True,
                               "result": "no tool call: " + (message.content or "")[:300]})
            if usage.tokens_in + usage.tokens_out > token_budget:
                return result("budget_exceeded")
            messages.append({"role": "user", "content": FINALIZE})
            try:
                forced, u = llm.chat(messages, schema=FinalAnswer.model_json_schema())
            except Exception as err:
                return result("error", f"LLM error: {err}")
            usage += u
            messages.append(forced)
            try:
                args = json.loads(forced.content)
            except json.JSONDecodeError:
                args = {}
            record = {"step": step, "name": "final_answer", "arguments": args, "error": False, "forced": True}
            tool_calls.append(record)
            done, observation = finalize(args, record)
            if done:
                return done
            record["result"] = observation[:300]
            messages.append({"role": "user", "content": observation})
            if usage.tokens_in + usage.tokens_out > token_budget:
                return result("budget_exceeded")
            continue

        for call in message.tool_calls:
            name, args = call.function.name, dict(call.function.arguments or {})
            record = {"step": step, "name": name, "arguments": args, "error": False}
            tool_calls.append(record)

            if name == "final_answer":
                done, observation = finalize(args, record)
                if done:
                    return done

            else:
                key = (name, json.dumps(args, sort_keys=True, default=str))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2:
                    record["result"] = "repeated call"
                    return result("stuck")
                if seen[key] == 2:          # warn first: the model may not have read the result
                    observation = f"You already called {name} with these arguments. Use the result above, " \
                                  "try different arguments, or call final_answer."
                else:
                    t = time.perf_counter()
                    out = run_tool(name, args)
                    record["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
                    record["error"] = bool(out.get("error"))
                    citable.update(out.get("citable", {}))
                    retrieved.extend(out.get("retrieved", []))
                    observation = out["content"]

            record.setdefault("result", observation[:300])
            messages.append({"role": "tool", "tool_name": name, "content": observation})

        if usage.tokens_in + usage.tokens_out > token_budget:
            return result("budget_exceeded")

    return result("max_iterations")
