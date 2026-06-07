"""
test_search.py — Person B
Tests for modules/search.py
"""

import pytest
from modules.search import build_index, search, is_ready

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "abstract": (
            "We propose a new simple network architecture, the Transformer, based "
            "solely on attention mechanisms, dispensing with recurrence and convolutions "
            "entirely. The model achieves superior quality on machine translation tasks."
        ),
        "url": "https://arxiv.org/abs/1706.03762",
        "year": 2017,
        "categories": ["cs.CL", "cs.LG"],
    },
    {
        "id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["He, K.", "Zhang, X."],
        "abstract": (
            "We present a residual learning framework to ease the training of deep "
            "neural networks. Residual networks won the ImageNet challenge with "
            "significantly lower error rates."
        ),
        "url": "https://arxiv.org/abs/1512.03385",
        "year": 2016,
        "categories": ["cs.CV"],
    },
    {
        "id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M."],
        "abstract": (
            "We introduce BERT, a new language representation model that pre-trains "
            "deep bidirectional representations by jointly conditioning on both left "
            "and right context in all layers."
        ),
        "url": "https://arxiv.org/abs/1810.04805",
        "year": 2019,
        "categories": ["cs.CL"],
    },
]


@pytest.fixture(autouse=True)
def fresh_index():
    """Build a fresh index before each test."""
    build_index(SAMPLE_PAPERS)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_is_ready_after_build():
    assert is_ready() is True


def test_search_returns_list():
    results = search("transformer attention mechanism")
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_result_has_required_fields():
    results = search("transformer")
    for paper in results:
        for field in ("id", "title", "authors", "abstract", "url", "year", "categories"):
            assert field in paper, f"Missing field: {field}"


def test_search_relevance_ordering():
    """The first result for 'transformer attention' should be more relevant than the last."""
    results = search("transformer attention", k=3)
    # The transformer paper should rank above the image recognition paper
    ids = [r["id"] for r in results]
    assert "1706.03762" in ids, "Attention paper should appear in transformer query results"
    assert ids.index("1706.03762") < ids.index("1512.03385") if "1512.03385" in ids else True


def test_search_respects_k():
    results = search("neural network", k=2)
    assert len(results) <= 2


def test_search_empty_query_returns_results():
    results = search("")
    assert isinstance(results, list)


def test_search_raises_if_index_not_built(monkeypatch):
    """search() must raise RuntimeError if index is None."""
    import modules.search as s
    original = s._index
    s._index = None
    with pytest.raises(RuntimeError, match="not yet built"):
        search("anything")
    s._index = original  # restore