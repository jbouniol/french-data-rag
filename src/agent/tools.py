"""Agent tools. Each tool validates its arguments, never raises on bad input it can explain,
and returns a dict: what the model reads + the article versions it may cite (num -> doc_id)."""
import datetime as dt
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

from src.retrieval.search import search

VERSIONS_PATH = "data/processed/article_versions.parquet"


def source_block(num: str, text: str, hierarchy: str = "", nota: str = "", extra: str = "") -> str:
    body = "\n".join(p for p in (hierarchy, text, f"Nota : {nota}" if nota else "") if p)
    body = body.replace("</source", "&lt;/source")    # the data cannot close its own tag
    return f'<source article="{num}"{extra}>\n{body}\n</source>'


# ============ 1. search_articles ============
class SearchArgs(BaseModel):
    query: str = Field(min_length=2, description="What to look for, in French, e.g. 'durée de la période d'essai apprenti'")
    k: int = Field(default=5, ge=1, le=8, description="Number of articles to return")


def search_articles(query: str, k: int = 5) -> dict:
    hits = search(query, k=k)
    text = "\n\n".join(source_block(h.metadata["num"], h.text, h.metadata["hierarchy"], h.metadata["nota"] or "")
                       for h in hits)
    return {"content": text or "No article found.",
            "citable": {h.metadata["num"]: h.doc_id for h in hits},
            "retrieved": [h.doc_id for h in hits]}


# ============ 2. get_article_at_date ============
class ArticleAtDateArgs(BaseModel):
    num: str = Field(description="Article number, e.g. 'L6222-7-1' or 'D6222-26'")
    date: dt.date = Field(description="Date in YYYY-MM-DD format")


@lru_cache(maxsize=1)
def load_versions() -> pd.DataFrame:
    return pd.read_parquet(VERSIONS_PATH)


def in_force_id(num: str) -> str:
    rows = load_versions()
    return rows[(rows.num == num) & (rows.etat == "VIGUEUR")].doc_id.iloc[0]


def normalize_num(num: str) -> str:
    """'l. 6222-7-1' / 'article L6222-7-1' -> 'L6222-7-1'."""
    return num.upper().replace("ARTICLE", "").replace(".", "").replace(" ", "")


def get_article_at_date(num: str, date: dt.date) -> dict:
    num = normalize_num(num)
    versions = load_versions()
    rows = versions[versions.num == num].sort_values("date_debut")
    if rows.empty:
        return {"content": f"Article {num} is not in the corpus.", "citable": {}, "retrieved": []}
    # date_fin is exclusive: the next version starts on that day
    match = rows[(rows.date_debut <= date) & (date < rows.date_fin) & (rows.etat != "ABROGE")]
    if match.empty:
        spans = ", ".join(f"{r.date_debut} to {r.date_fin} ({r.etat})" for r in rows.itertuples())
        return {"content": f"No version of article {num} was in force on {date}. Known versions: {spans}.",
                "citable": {}, "retrieved": []}
    r = match.iloc[0]
    until = "today" if r.date_fin.year == 2999 else str(r.date_fin)
    extra = f' version="{r.doc_id}" in_force_from="{r.date_debut}" in_force_until="{until}"'
    return {"content": source_block(num, r.text, r.hierarchy, r.nota or "", extra),
            "citable": {num: r.doc_id}, "retrieved": [r.doc_id]}


# ============ 3. compute_remuneration ============
# D6222-26 in force since 2019-01-01: % of the SMIC by age band, for contract years 1, 2, 3
GRID = {"16-17 ans": (27, 39, 55), "18-20 ans": (43, 51, 67), "21-25 ans": (53, 61, 78), "26 ans et plus": (100, 100, 100)}
GRID_SINCE = dt.date(2019, 1, 1)

# The SMIC amount is NOT in the Code du travail: an external, dated parameter (decree published each year).
# Monthly gross amount for 35 h/week. Source: service-public.fr / decrees on the SMIC revaluation.
SMIC_MONTHLY_GROSS = {dt.date(2024, 11, 1): 1801.80, dt.date(2026, 1, 1): 1823.03}


class RemunerationArgs(BaseModel):
    age: int = Field(ge=14, le=99, description="Age of the apprentice in years")
    contract_year: int = Field(ge=1, le=5, description="Year of execution of the contract: 1, 2, 3 (or more if extended)")
    licence_pro: bool = Field(default=False, description="True if the apprentice prepares a licence professionnelle")
    on_date: dt.date | None = Field(default=None, description="Date of the pay period, YYYY-MM-DD (default: today)")


def smic_at(day: dt.date) -> tuple[dt.date, float] | None:
    known = [d for d in SMIC_MONTHLY_GROSS if d <= day]
    return (max(known), SMIC_MONTHLY_GROSS[max(known)]) if known else None


def compute_remuneration(age: int, contract_year: int, licence_pro: bool = False, on_date: dt.date | None = None) -> dict:
    day = on_date or dt.date.today()
    if day < GRID_SINCE:
        return {"content": f"The grid used here applies from {GRID_SINCE}. For {day}, read the version of D6222-26 "
                           "in force on that date with get_article_at_date.", "citable": {}, "retrieved": []}
    applied, notes = ["D6222-26"], []
    if age < 16:
        applied.append("D6222-27")
        notes.append("under 16: same pay as 16-17 year olds (D6222-27)")
    band = "16-17 ans" if age <= 17 else "18-20 ans" if age <= 20 else "21-25 ans" if age <= 25 else "26 ans et plus"
    year = contract_year
    if year > 3:
        applied.append("D6222-28")
        notes.append(f"contract year {year}: an extended apprenticeship keeps the rate of the last year (D6222-28)")
        year = 3
    if licence_pro:
        applied.append("D6222-32")
        notes.append("licence professionnelle: rate of the 2nd year of the contract (D6222-32)")
        year = 2
    percent = min(GRID[band][year - 1], 100)          # D6222-31: never above 100 % of the SMIC
    if band in ("21-25 ans", "26 ans et plus"):
        notes.append("or the conventional minimum wage of the job if it is higher (collective agreement: not covered)")

    lines = [f"Minimum pay (D6222-26): {percent} % of the SMIC for an apprentice aged {age} ({band}), "
             f"contract year {contract_year}."]
    smic = smic_at(day)
    if smic:
        smic_date, smic_amount = smic
        lines.append(f"SMIC on {day}: {smic_amount:.2f} EUR gross per month (35 h), in force since {smic_date} "
                     f"(external parameter, not in the Code du travail). "
                     f"Minimum: {percent} % x {smic_amount:.2f} = {smic_amount * percent / 100:.2f} EUR gross per month.")
    lines += [f"Note: {n}." for n in notes]
    return {"content": "\n".join(lines),
            "citable": {num: in_force_id(num) for num in applied},
            "retrieved": [],
            "percent_smic": percent}


# ============ registry ============
TOOLS = {
    "search_articles": (search_articles, SearchArgs,
                        "Search the Code du travail articles (apprenticeship, professionalisation contract, young "
                        "workers, paid leave, trial period, contract termination). Returns the best articles in force."),
    "get_article_at_date": (get_article_at_date, ArticleAtDateArgs,
                            "Get the version of one article that was in force on a given date "
                            "(for questions about the past, e.g. 'en 2018')."),
    "compute_remuneration": (compute_remuneration, RemunerationArgs,
                             "Compute the legal minimum pay of an apprentice (D6222-26 grid, % of the SMIC and "
                             "amount in euros) from age and year of contract."),
}


def run_tool(name: str, arguments: dict) -> dict:
    """Validate the arguments with Pydantic, then execute. Errors come back as results, never as crashes."""
    if name not in TOOLS:
        return {"content": f"Tool error: unknown tool '{name}'. Available: {', '.join(TOOLS)}, final_answer.",
                "error": True}
    fn, schema, _ = TOOLS[name]
    try:
        args = schema.model_validate(arguments)
    except Exception as err:
        return {"content": f"Tool error: invalid arguments for {name}: {err}", "error": True}
    return fn(**args.model_dump())
