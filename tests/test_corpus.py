"""Check the files produced by `uv run python -m src.ingest.build`."""
from datetime import date

import pandas as pd
import pytest

from src.ingest.build import parse_hierarchy

ARTICLES = "data/processed/articles.parquet"
VERSIONS = "data/processed/article_versions.parquet"


@pytest.fixture(scope="module")
def articles() -> pd.DataFrame:
    return pd.read_parquet(ARTICLES)


@pytest.fixture(scope="module")
def versions() -> pd.DataFrame:
    return pd.read_parquet(VERSIONS)


def test_article_count(articles):
    assert 400 <= len(articles) <= 600


def test_no_duplicate_doc_id(articles):
    assert articles.doc_id.is_unique


def test_all_in_force(articles):
    assert (articles.etat == "VIGUEUR").all()
    assert articles.is_in_corpus.all()


def test_dates_are_parsable(articles, versions):
    for df in (articles, versions):
        assert df.date_debut.map(lambda d: isinstance(d, date)).all()
        assert df.date_fin.map(lambda d: isinstance(d, date)).all()
        assert (df.date_debut <= df.date_fin).all()


def test_no_empty_text(articles):
    assert (articles.text.str.strip().str.len() > 0).all()


def test_versions_cover_corpus(articles, versions):
    assert set(articles.doc_id) <= set(versions.doc_id)
    assert set(versions.num) == set(articles.num)
    assert versions[versions.is_in_corpus].doc_id.is_unique


def test_parse_hierarchy_keeps_last_duplicate_level():
    subtitles = (
        "Partie législative   - Sixième partie : La formation  - Livre III : A  "
        "- Livre III : B - Titre II : T  - Chapitre V : C - Section 1 : S."
    )
    assert parse_hierarchy(subtitles) == (
        "Partie législative > Sixième partie : La formation > Livre III : B > Titre II : T > Chapitre V : C"
    )
