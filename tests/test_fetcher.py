import pytest
from modules.fetcher import fetch_papers, fetch_paper_by_id

REQUIRED_KEYS = {"id", "title", "authors", "abstract", "url", "year", "categories"}


def test_fetch_papers_returns_list():
    results = fetch_papers("transformer attention", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_fetch_papers_paper_shape():
    results = fetch_papers("BERT language model", max_results=2)
    for paper in results:
        assert REQUIRED_KEYS.issubset(paper.keys()), f"Missing keys in: {paper.keys()}"
        assert isinstance(paper["id"], str) and len(paper["id"]) > 0
        assert isinstance(paper["title"], str) and len(paper["title"]) > 0
        assert isinstance(paper["abstract"], str) and len(paper["abstract"]) > 20
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["year"], int) and paper["year"] > 1990
        assert paper["url"].startswith("https://arxiv.org/abs/")


def test_fetch_paper_by_id():
    paper = fetch_paper_by_id("1706.03762")
    assert paper is not None
    assert REQUIRED_KEYS.issubset(paper.keys())
    assert "attention" in paper["title"].lower() or "transformer" in paper["title"].lower()


def test_fetch_paper_by_id_not_found():
    paper = fetch_paper_by_id("0000.00000")
    assert paper is None or isinstance(paper, dict)