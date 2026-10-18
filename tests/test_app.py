"""Tests for the Alter-Ego Daylight Keynote Web Demo backend (`app/server.py`)."""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import threading
import urllib.request

from app.server import (
    AlterEgoRequestHandler,
    DEMO_SCENARIOS,
    enrich_trace_payload,
    get_article_timeline,
    get_system_status,
    load_golden_and_eval,
    resolve_article,
)


def test_resolve_article_by_id_and_num():
    by_num = resolve_article("L6222-35")
    assert by_num is not None
    assert by_num["num"] == "L6222-35"
    assert by_num["doc_id"] == "LEGIARTI000021342274"
    assert by_num["etat"] == "VIGUEUR"

    by_id = resolve_article("LEGIARTI000038033238")
    assert by_id is not None
    assert by_id["num"] == "D6222-26"
    assert by_id["etat"] == "VIGUEUR"


def test_article_timeline_detects_repeal_gap():
    tl = get_article_timeline("D6222-33")
    assert tl["found"] is True
    gap_nodes = [n for n in tl["timeline"] if n["type"] == "gap"]
    assert len(gap_nodes) == 1
    assert gap_nodes[0]["date_debut"] == "2019-01-01"
    assert gap_nodes[0]["date_fin"] == "2020-04-01"


def test_enrich_trace_payload_verifies_citations_and_gates():
    sc = DEMO_SCENARIOS[0]
    enriched = enrich_trace_payload(
        question=sc["question"],
        mode=sc["mode"],
        trace_dict=sc["instant_trace"],
        source_type="instant_stage",
    )
    assert enriched["stop_reason"] == "answered"
    assert len(enriched["sources"]) >= 2
    cited_nums = {s["num"] for s in enriched["sources"] if s["is_cited"]}
    assert {"D6222-26", "L6222-35"}.issubset(cited_nums)
    assert enriched["gates"]["xml_isolation"]["passed"] is True
    assert enriched["gates"]["constrained_json"]["passed"] is True
    assert enriched["gates"]["symbolic_verification"]["passed"] is True


def test_system_status_and_golden_loader():
    status = get_system_status()
    assert status["corpus"]["vigueur_articles"] == 540
    assert status["corpus"]["total_versions"] == 1066
    assert len(status["scenarios"]) == len(DEMO_SCENARIOS)

    golden = load_golden_and_eval()
    assert golden["total"] == 40
    assert len(golden["questions"]) == 40


def test_http_endpoints_roundtrip():
    server = ThreadingHTTPServer(("127.0.0.1", 0), AlterEgoRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # 1. Index HTML
        with urllib.request.urlopen(f"{base}/") as resp:
            html = resp.read().decode("utf-8")
            assert "Alter-" in html
            assert "Know your" in html

        # 2. /api/status
        with urllib.request.urlopen(f"{base}/api/status") as resp:
            st = json.loads(resp.read().decode("utf-8"))
            assert st["corpus"]["vigueur_articles"] == 540

        # 3. /api/ask (instant scenario)
        req = urllib.request.Request(
            f"{base}/api/ask",
            data=json.dumps({"instant_scenario_id": "sarah_multi_tool"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            ask_data = json.loads(resp.read().decode("utf-8"))
            assert ask_data["stop_reason"] == "answered"
            assert "LEGIARTI000038033238" in ask_data["citations"]

        # 4. /api/tools/remuneration
        req_rem = urllib.request.Request(
            f"{base}/api/tools/remuneration",
            data=json.dumps({"age": 21, "contract_year": 2, "on_date": "2026-03-01"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_rem) as resp:
            rem_data = json.loads(resp.read().decode("utf-8"))
            assert rem_data["percent_smic"] == 61
            assert rem_data["monthly_gross_eur"] == 1112.05

        # 5. /api/tools/article-at-date (repeal gap)
        req_hist = urllib.request.Request(
            f"{base}/api/tools/article-at-date",
            data=json.dumps({"num": "D6222-33", "date": "2019-06-15"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_hist) as resp:
            hist_data = json.loads(resp.read().decode("utf-8"))
            assert hist_data["in_gap_or_missing"] is True
    finally:
        server.shutdown()
        server.server_close()
