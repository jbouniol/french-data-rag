"""Deterministic agent tools: no LLM needed."""
from datetime import date

import pytest

from src.agent.tools import compute_remuneration, get_article_at_date, normalize_num, run_tool


# ============ compute_remuneration (D6222-26 grid) ============
@pytest.mark.parametrize("age, year, percent", [
    (16, 1, 27), (17, 3, 55),          # 16-17
    (18, 1, 43), (20, 2, 51),          # 18-20
    (21, 1, 53), (25, 3, 78),          # 21-25
    (26, 1, 100), (30, 3, 100),        # 26 and over
])
def test_grid(age, year, percent):
    assert compute_remuneration(age, year, on_date=date(2026, 3, 1))["percent_smic"] == percent


def test_under_16_paid_like_16_17():
    r = compute_remuneration(15, 2, on_date=date(2026, 3, 1))
    assert r["percent_smic"] == 39 and "D6222-27" in r["citable"]


def test_extended_contract_keeps_last_year_rate():
    r = compute_remuneration(19, 4, on_date=date(2026, 3, 1))
    assert r["percent_smic"] == 67 and "D6222-28" in r["citable"]


def test_licence_pro_uses_second_year():
    r = compute_remuneration(22, 1, licence_pro=True, on_date=date(2026, 3, 1))
    assert r["percent_smic"] == 61 and "D6222-32" in r["citable"]


def test_amount_uses_dated_smic():
    r = compute_remuneration(18, 1, on_date=date(2026, 3, 1))
    assert f"{1823.03 * 0.43:.2f}" in r["content"]
    assert r["citable"]["D6222-26"] == "LEGIARTI000038033238"
    assert r["retrieved"] == ["LEGIARTI000038033238"]


def test_before_grid_redirects_to_dated_article():
    r = compute_remuneration(18, 1, on_date=date(2017, 6, 1))
    assert "get_article_at_date" in r["content"] and not r["citable"]


# ============ get_article_at_date ============
def test_version_in_force_today():
    r = get_article_at_date("D6222-26", date(2026, 1, 1))
    assert r["citable"] == {"D6222-26": "LEGIARTI000038033238"}


def test_old_version():
    r = get_article_at_date("D6222-26", date(2017, 6, 1))
    assert r["citable"] == {"D6222-26": "LEGIARTI000032458704"}
    assert "25 %" in r["content"]


def test_boundary_date_belongs_to_new_version():
    r = get_article_at_date("D6222-26", date(2019, 1, 1))
    assert r["citable"] == {"D6222-26": "LEGIARTI000038033238"}


def test_gap_returns_no_article():
    # D6222-33 repealed on 2019-01-01, back in force on 2020-04-01
    r = get_article_at_date("D6222-33", date(2019, 6, 1))
    assert r["citable"] == {} and "No version" in r["content"]


def test_unknown_article():
    assert get_article_at_date("L9999-99", date(2026, 1, 1))["citable"] == {}


def test_normalize_num():
    assert normalize_num("article l. 6222-7-1") == "L6222-7-1"
    assert normalize_num("Art. L. 6222-7-1 du Code du travail") == "L6222-7-1"
    assert normalize_num("[D6222-26]") == "D6222-26"


# ============ run_tool: validation before execution ============
def test_unknown_tool_is_an_error_not_a_crash():
    assert run_tool("google_it", {"query": "x"})["error"]


def test_invalid_arguments_are_an_error_not_a_crash():
    assert run_tool("compute_remuneration", {"age": "dix-huit", "contract_year": 1})["error"]
    assert run_tool("get_article_at_date", {"num": "D6222-26", "date": "hier"})["error"]


def test_valid_call_with_string_date():
    r = run_tool("get_article_at_date", {"num": "D6222-26", "date": "2017-06-01"})
    assert r["citable"] == {"D6222-26": "LEGIARTI000032458704"}
