import pytest
from modules.compare import compare
import json, pathlib

PAPERS = json.loads(
    pathlib.Path("mock_data/sample_papers.json").read_text()
)

REQUIRED_KEYS = {"query", "tfidf", "transformer", "overlap_count", "overlap_ids"}


def test_compare_returns_correct_shape():
    result = compare("transformer attention", PAPERS)
    assert REQUIRED_KEYS.issubset(result.keys())


def test_compare_tfidf_results_non_empty():
    result = compare("transformer attention", PAPERS)
    assert isinstance(result["tfidf"]["results"], list)
    assert len(result["tfidf"]["results"]) > 0


def test_compare_overlap_count_is_int():
    result = compare("BERT language", PAPERS)
    assert isinstance(result["overlap_count"], int)
    assert result["overlap_count"] >= 0


def test_compare_top_terms_are_strings():
    result = compare("neural network", PAPERS)
    for term in result["tfidf"]["top_terms"]:
        assert isinstance(term, str)


def test_compare_empty_papers():
    result = compare("anything", [])
    assert result["tfidf"]["results"] == []
    assert result["overlap_count"] == 0